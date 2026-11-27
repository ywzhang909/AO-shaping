# Agent 当日记录 2026-11-27（beam quality 整形链路降本，离线）

本轮**未占用硬件**；实验验证清单见文末，须用户确认后再执行。
①「隐藏页停渲染」已提交 `f43b4bc8`，其硬件 pytest 固化见 `19e88928`。

## 1. ② 逐帧计算降本（离线已改并自检）

只改 `src/ao_shaping/gui/ccd/beam_quality/analyzer.py`：

1. `_background_estimate`：≥ 20 万像素的帧按 4 像素步进取中值；小帧仍走精确全幅中值。
2. `_largest_component`：只在非零像素包围盒内做 `ndimage.label`，再还原成全幅布尔图；
   盒外全为背景，连通关系与全幅标注一致。
3. `fit_profile`：给 `curve_fit` 传解析雅可比 `_gaussian_1d_jac`，保留 `maxfev=4000`
   的失败兜底口径。

合成帧（640×480 / 1344×760 / 斜坡背景 / 饱和帧）离线对比，中位数：

| 场景 | 改前 ms | 改后 ms | 提速 |
| --- | ---: | ---: | ---: |
| 640×480 平背景 | 12.85 | 6.95 | 46.0% |
| 1344×760 平背景 | 30.86 | 16.50 | 46.5% |
| 1344×760 斜坡背景 | 36.77 | 25.57 | 30.5% |
| 640×480 饱和 | 12.97 | 6.97 | 46.2% |

数值差异：拟合参数 ≤ 5e-9，`center_spread` ≤ 0.025 px；未改变质心/椭圆口径。

自检：`.venv\Scripts\ruff.exe check` all passed；
`pytest tests/ao_shaping/gui/ccd/ -q --import-mode=importlib` 102 passed / 1 skipped；
`pytest tests/ao_shaping/gui/slm/test_transfer_desktop_live_view.py -q --import-mode=importlib`
24 passed；`git diff --check` 针对改动文件 exit 0。

## 2. GPU 批量可行性结论（离线实测，不采用）

`torch` 2.14+cu132（1 GPU）与 `cupy` 14.2 均可用。逐帧 reduction / elementwise
算子搬 GPU 后，单帧含 H2D+D2H 的收益很小；把 stats 融合成 bundle 后，
计入传输的批量均值在 640×480 为 n=4/8/16 → 2.00/1.53/1.18 ms/帧，
1344×760 为 4.25/3.30/3.28 ms/帧；n=32 不再改善。

- 单帧：640×480 bundle CPU 3.55 ms vs GPU 含传输约 2.85 ms；1344×760 CPU
  10.32 ms vs GPU 约 2.28 ms，但这是纯 stats，不含 `ndimage.label` / `curve_fit` /
  `to_uint8`——后者才是主热点且不是 GPU 批处理候选。
- 30 Hz 下缓冲 8–16 帧等于新增 0.27–0.53 s 延迟，与实时整形目标冲突。
- `torch.median` 对偶数个元素取较小中间值（探针中差 13 counts），`torch.quantile`
  与 `np.median` 一致但更慢；两种都不能无损替换现有中值口径。
- **裁定**：不采用 GPU 批量；继续用 CPU 抽样中值 + ROI 标注 + 解析雅可比。
  原始基准脚本与日志留在临时目录
  `C:\Users\zhangh\AppData\Local\Temp\opencode\gpu_batch_bench\`。

## 3. 实验清单（待用户确认，本轮未执行）

1. 设备：占用 SLM #1 + 大恒 CCD cam 0（页 1 真实闭环）；页 0 仿真源不占相机。
   开始前、结束后须在回复中明确报告占用与释放。
2. 命令：

   ```bash
   .venv\Scripts\python.exe -m pytest tests/ao_shaping/gui/ccd/test_beam_quality_hardware.py --hardware -v -q --import-mode=importlib
   ```

3. 观察：`visible` / `hidden` 两段的 `cpu_per_wall`、`processed_per_wall`、
   `poll_interval_ms`、`analyze_ms_median`；运行中 checkpoint 仍须 `render_active=false`；
   页 1 会话 `journal.json` / `optical_probe.json` 存在，退出后 `beam_source=None`、
   `transfer_running=false`。
4. 证据：`data/debug/beam_quality/hidden_gate_hardware.json`；会话目录
   `data/debug/slm_transfer/<时间戳>/`。
5. 若失败：先看 `current_problems()` 与运行按钮状态，不要重复启动第二个占设备的进程。

## 4. §5 数据缺口：采集/重建代码已离线备好（未实验）

来源：`report/shape_trials/closed_loop_speed_tsne_20261125/closed_loop_speed_analysis_20261125.md` §5。

1. **逐轮分项耗时（原缺口 1）**：新增 `scripts/slm/collect_closed_loop_timing_20261127.py`，
   不改被别的车道改写的 `closed_loop_shaping.py` / `closed_loop_shaping_page.py`。
   - 通过 `ClosedLoopRunner` 的 `on_event` / `on_frame` 钩子 + `TimingBench` 装饰器采集；
     事件对口径：`适配重拟合`→`adaptation`=fit，`adaptation`→`生成第 N 轮候选`=adapt，
     `生成第 N 轮候选`→`下发第 N 轮候选`=candidate，`apply_phase`+`capture`=capture。
   - 帧回调**当场**采样 `apply_phase_ms` / `ccd_capture_ms`，避免末轮值覆盖前几轮；
     门禁失败的 baseline 也由 `round` 事件兜底写入 `rounds.jsonl`。
   - 默认 plan-only：不导入设备模块、不加锁、不建输出目录；`--execute` 才占设备，
     `DeviceLock.default()` + `--wait-until-idle`，`finally` 释放并记 `lock` 段。
2. **权重轨迹（原缺口 2，已离线补齐）**：新增
   `scripts/slm/reconstruct_debug_weights_20261127.py`。原会话 `20261124_200949`
   的 `debug_weights=false` 且 `save_dataset=false`，但 `fit/iterations.pkl` 有完整
   `parameters`；脚本复用 `fit._flatten_parameters` / `_write_weight_trajectory`，
   写出 `data/debug/slm_transfer/20261124_200949/fit/debug_weights_reconstructed/`
   下四个产物 + `verify.json`（7 迭代、17 键，npz 首末值自检通过）。
   `reconstruction.json` 明确 `hardware_used=false`、`refit_performed=false`，
   **是重建不是新采集**，原 `debug_weights/` 目录拒绝覆盖。
3. **溯源登记（原缺口 3，仍阻塞）**：`scripts/_common/provenance.py::REPORTS` 正被另一车道
   整文件改写，仍不碰；待其提交后再补登记并跑 `sync_report_provenance.py --check`。

自检：`.venv\Scripts\python.exe -m compileall` 两脚本通过；`ruff check` 四个新文件 all passed；
`pytest tests/ao_shaping/scripts/test_collect_closed_loop_timing_20261127.py
tests/ao_shaping/scripts/test_reconstruct_debug_weights_20261127.py -q --import-mode=importlib`
**11 passed**（含 fake bench 端到端多轮落盘、gate-fail 兜底落盘、plan-only 不建目录）；
针对四个新文件 `git diff --check` exit 0。

## 5. §5 采集器实验清单（待用户确认，本轮未执行）

1. 设备：SLM #1 + 大恒 CCD cam 0；开始前、结束后在回复中明确报告占用与释放。
   上次成功关闭 SLM #1 为 2026-11-27 16:02:14，当前已过 >200 s 冷却余量，
   仍按 `--wait-until-idle 300` 让设备锁裁决，不无条件抢锁。
2. 先跑离线复核（不占设备）：

   ```bash
   .venv\Scripts\python.exe scripts/slm/collect_closed_loop_timing_20261127.py --out data/debug/closed_loop_timing_20261127 --checkpoint data/debug/slm_transfer/ab_session_20261124/refit_forward/frozen_parameters.json --session-dir data/debug/closed_loop_timing_20261127 --source-dataset data/debug/slm_transfer/ab_session_20261124/capture
   ```

   确认 stdout 的 `config.candidate_policy=auto`、device 为 SLM #1 / daheng / cam 0。
3. 光路：按 `libs/skills/ao-optical-path-calibration/SKILL.md` 六步探测，
   确认 `光路有光`、曝光不超过 CCD 量程、SLM 相位自检通过；页 1 会话必须
   `page.set_camera("daheng")`，本脚本已直接用 `HardwareBench(camera_type="daheng", cam_id=0)`。
4. 再执行（占设备）：

   ```bash
   .venv\Scripts\python.exe scripts/slm/collect_closed_loop_timing_20261127.py --out data/debug/closed_loop_timing_20261127 --execute --checkpoint data/debug/slm_transfer/ab_session_20261124/refit_forward/frozen_parameters.json --session-dir data/debug/closed_loop_timing_20261127 --source-dataset data/debug/slm_transfer/ab_session_20261124/capture --max-rounds 8 --step-iterations 60 --exposure-ms 0.05 --centre-x 1228 --centre-y 1107
   ```

   注意：`analysis_grid` 直接在原始 CCD 帧上采样，中心必须是光斑质心，不能用页面
   GUI 默认的 (500,500)（那是 capture_grid 缩放坐标）。已据 18:50 光路探针实测质心
   (1228,1107) 与 frozen origin (1229.20,1106.23) 把 `--centre-*` 默认值改为质心；
   默认曝光由 20 ms 改为探针建议的 0.05 ms（8-bit 峰值约 117，0.12 ms 时 197 已超
   0.7×满量程 178、暖机有饱和风险）。若门禁 `low_signal` 先查光路/曝光，不盲目放大。
5. 证据：`data/debug/closed_loop_timing_20261127/session.json`（锁、设备回读、时间线、
   退出码）与 `rounds.jsonl`（每轮 `fit_ms` / `adapt_ms` / `candidate_ms` /
   `apply_phase_ms` / `ccd_capture_ms` / `capture_ms` / gate / 分数 / 决策）。
6. 收尾：确认 `lock.released=true`、`release_error=null`，SLM 恢复正常平相位、CCD 关闭，
   日志末次「成功关闭SLM #1」时间戳；失败时先看 `session.json.reason` 与 gate 原因，
   不重复启动第二个占设备的进程。

### 5.1 光路探针（18:50，通过）

`.venv\Scripts\python.exe scripts/slm/probe_optical_path_validity.py --session closed_loop_timing_gate_20261127 --exposures 0.05,0.12,0.2,0.5,1 --defocus-amps 0,0.5,1.0,-1.0 --bit-depth 8 --no-reference --no-plot --wait-until-idle 300`
判定 ok：8-bit Mono8，曝光 0.05/0.12 ms 可用，SLM 离焦峰值变化 29.0%、宽度 24.8%
（确认真在调制），光斑质心 (1228,1107)；0.05 ms 峰值约 117（远低于 0.7×255=178），
0.12 ms 峰值 197 已超阈——故生产曝光锁定 0.05 ms。证据
`data/debug/optical_path_probe/closed_loop_timing_gate_20261127/20261127_105051Z/optical_path_probe.json`；
18:52:17 成功关闭 SLM #1。

### 5.2 采集执行：两次失败根因与修复（均已提交）

- 第 1 次（19:01，`data/debug/closed_loop_timing_20261127/`，`ff657cf1` 前）：baseline r00
  gate_ok、score=1.0126，第 1 轮候选生成时 fault：
  `The size of tensor a (128) must match ... b (640) at dimension 2`。根因：采集器
  `--grid-size` 默认 128 生成 128×128 start_phase，但冻结 checkpoint
  `refit_forward/frozen_parameters.json` 的 `grid_size=640`，`FrozenDefocusForward` 按 640
  建 panel_grid，`forward.predict` 严格要 640×640。修复 `9434408`：新增 `checkpoint_grid_size()`
  从 checkpoint 读 `grid_size` 自动对齐 start_phase，`--grid-size` 退化为无检查点兜底；
  plan_doc 上报 `grid_size=640`（生效）与 `grid_size_cli=128`（原输入）。`ff657cf1` 另把
  `source_dataset` 目录契约（须含 `manifest.json`）、`reference_start` 与负 ramp 校验前移到
  占设备之前，避免占机后才抛 `missing negative ramp`。
- 第 2 次（19:14，`data/debug/closed_loop_timing_20261127_rerun/`）：shape 失配已解，baseline
  正常下发 640×640 相位，但第 1 轮候选报 device 失配 `cuda:0 and cpu`。根因：runner 的
  `refine_phase` 调用点不传 device（init/target 在 CPU），而 standalone 采集器构造
  `OnlineTransferAdapter` 未传 device → `FrozenDefocusForward` 默认 cuda。修复 `ff20cc1`：
  采集器 adapter 工厂显式 `device="cpu"`（与离线端到端验证组合一致），plan_doc 记录该值。
  两次均确认 `lock.released=true`、`release_error=null`、SLM 正常关闭、无残留硬件进程。

离线门禁（修复后）：ruff 两文件 exit 0；`pytest ... -q --import-mode=importlib` **15 passed**；
targeted `git diff --check` exit 0；真实 checkpoint 640 网格 `refine_phase` dry-run exit 0；
真实 adapter(device=cpu)+真实 forward+假光斑 bench 端到端 4 轮 state=stopped（固化进测试）。

### 5.3 采集执行：第 3 次成功（19:18–19:20）

命令同 §5 实验清单、输出目录改为 `data/debug/closed_loop_timing_20261127_rerun2/`。
`session.json`：`state=stopped`、`reason=连续 5 轮候选未被接受，已停止`、`exit_code=0`、
`rounds_recorded=7`（baseline r00 + 6 轮候选）；`lock.acquired=11:18:52Z`、`released=11:20:18Z`、
`held_s=85.256`、`released=true`、`release_error=null`；19:20:18 成功关闭 SLM #1，无残留进程。
`config` 生效值：`grid_size=640`（自动对齐）、`exposure_ms=0.05`、`centre_xy=(1228,1107)`、
`candidate_policy=auto`、`step_iterations=60`。

`rounds.jsonl` 判读：7 轮全部 `gate_ok=true`（无饱和/低信号）；`capture_ms` 约 750–1573 ms
（其中 `ccd_capture_ms` 291–326 ms）；`apply_phase_ms` 约 459–1281 ms；6 轮 `candidate_ms`
均有值（r01 12426.9 ms 含首次适配构建，r02–r06 约 5.8–8.4 s）。**`fit_ms` / `adapt_ms` /
`adaptation` 全为 null 是真实语义**：`auto` 策略下仅 r01 被接受（measured 1.168 < applied
1.260），`_pending_since_refit=1 < refit_every=2`，`adapter.ready()` 为假 → 全程未触发
`适配重拟合` 事件，无对应事件对可拆；不是采集缺字段。baseline score=1.2605，末轮 1.2145，
连续 5 轮（r02–r06）实测未改善被拒。

**§5 缺口收敛**：缺口 1（逐轮 capture/candidate 分项）已由本场多轮 rounds.jsonl 补齐；
fit/adapt 分项因本场适配门槛未达而无数据，若要覆盖需在 `candidate_policy=manual` 或放宽
`refit_every` 下再跑一场（本次不改他车道 runner，保持默认策略的真实性）。缺口 2 已离线重建
补齐。缺口 3 溯源登记仍待 `scripts/_common/provenance.py` 车道提交。

## 6. ② 逐帧降本实验验证（19:32–19:35，通过）

用户裁定「进入 02 逐帧降本实验验证」后执行 §3 清单命令：

```bash
.venv\Scripts\python.exe -m pytest tests/ao_shaping/gui/ccd/test_beam_quality_hardware.py --hardware -v -q --import-mode=importlib
```

结果 **1 passed in 155.28 s**（默认不带 `--hardware` 为 1 skipped，门禁正常）。这是 ① 停渲染 +
② analyze CPU 降本的组合验收：可见段 `analyze_ms_median` 非空、`render_active=True`；
隐藏段 `render_active=False`、`analyze_calls < visible`、`cpu_per_wall < visible`、
`trend_len > visible`（断言 289–300 全过）。闭环段 `beam_render_active=False`、
`poll_interval_ms=1000`（降频保持），`timed_out=False`、状态含「会话完成」、
`journal.json` / `optical_probe.json` 均存在。会话目录 `data/debug/slm_transfer/20261127_193251`。

**设备占用 / 释放**：占用 19:32:29 SLM #1 + Daheng cam_id=0（页 1 真实闭环），页 0 仿真源不占
相机；释放 19:34:59 `close_wall_s=0.518`、`beam_source=None`、`transfer_running=False`，
证据 `data/debug/beam_quality/hidden_gate_hardware.json`（`devices_occupied` /
`devices_released` 两段齐全），无残留 pytest/SLM 进程。开始前已过 §5 采集关闭（19:20:18）
后 642 s 冷却。同期检测到其他在飞车道进程（`diagnose_zero_phase_exposure` /
`test_report_provenance`），与本会话无关、未干预。

说明：测试把 `visible` / `hidden` 采样用于断言而未写进证据 JSON（非本次故障），
故 `analyze_ms_median` 的具体数值只在该次进程的断言中校验；离线合成帧的改前/改后数值对比
见本文 §1（`analyzer.py` 已提交 `8e5116f5`）。

## 7. A+B：在线重拟合与拒轮样本登记（20:01–20:23）

用户裁定「A + B 都做（先 A 验证再 B 深化）」。A = 采集器暴露 `--refit-every` 触发在线重拟合；
B = 门禁通过但被拒的候选轮也登记为适配样本（runner 增默认关闭开关）。

### 7.1 实现与离线门禁

- `acfabbc6`：采集器 `--refit-every` + `AdaptationPolicy`，plan_doc 上报。
- `ac883106`：补 `test_export_checkpoint_handles_missing_registration`（提交时误吞他车道 224 行
  删除，已用临时 index 重建提交修正，他车道删除仍是 worktree dirty）。
- `97f8f46d`：`_export_checkpoint` 中 `registration_value` 是 `(dy, dx) | None`，原
  `float(...)` 在 `None` 时抛 TypeError；改为按 `fit.py:574` schema 写 list 或 None。
  仅提交 1 个 hunk，他车道 runner 18 处 dirty 未并吞。
- `66f37e4`：`--register-rejected-samples` 采集器 CLI + plan/config 透传。
- `da2c22b1`：runner `ClosedLoopConfig.register_rejected_samples=False` 默认关闭；开启后在
  reject 分支把 (候选相位, 该轮实测帧) 登记为适配样本，下发仍回滚到已确认相位。
  仅提交 3 个纯新增 hunk，他车道 runner/test 的 dirty 删除仍留 worktree。

离线门禁：两轮 `pytest tests/ao_shaping/gui/slm/test_closed_loop_shaping.py
tests/ao_shaping/scripts/test_collect_closed_loop_timing_20261127.py -q --import-mode=importlib`
分别 47 passed / 49 passed；ruff 与定向 `git diff --check` 均 exit 0；plan-only 显示
`refit_every=1`、`register_rejected_samples=true`。

### 7.2 三次实验判读

1. 19:51–19:53（`data/debug/closed_loop_timing_20261127_refit1/`）：`--refit-every 1`
   第 2 轮触发适配，但 40 s 后 `state=fault`：`float() argument ... not 'NoneType'`。
   根因即 §7.1 的 `registration_value=None`；写完空目录 `promoted_0015` 后抛错。
   `lock.released=true`、`release_error=null`，SLM 19:53:17 关闭。
2. 20:02–20:03（`..._refit1_rerun/`）：修复后 `state=stopped`、`exit_code=0`、9 轮，但
   **7 个 fit/adapt 字段仍全 null**：首个被接受轮 r03 恰是 `validation_every=3` 的验证留一，
   seed 只有 3 行 train（`min_train_rows=4`），`ready()` 始终为假；B 正是针对该饥饿。
   `lock.released=true`，SLM 20:03:44 关闭。
3. 20:13–20:22（`..._refit1b/`，`--refit-every 1 --register-rejected-samples`）：
   `state=stopped`、`exit_code=0`、9 轮；manifest 22 行，含 `rejected_round_02/04/05/06/07/08`；
   **7 次 `适配重拟合` → `adaptation=promoted`**，生成 `promoted_0015`…`promoted_0021`。
   `fit_ms` 38.6–77.8 s（均值 60.2 s），`adapt_ms` 1.23–1.74 s（均值 1.51 s）。
   baseline 1.1906；最佳 r03 1.1035（-7.3%）；但 r04–r08 实测回升到 1.3513（+13.5%），
   说明**内部验证晋级没有跟上实测均匀度**：连续 promotion 后预测模型反而把相位带偏。
   `lock.acquired=12:13:44Z`、`released=12:22:40Z`、`held_s=536.287`、
   `released=true`、`release_error=null`，SLM 20:22:40 正常关闭。

**设备占用 / 释放**：§7 三场均为 SLM #1 + Daheng cam_id=0，页 0 仿真源不占相机；三场均
确认锁释放、无残留采集进程。实验前已核无在飞 `.venv` Python 进程、锁文件时间陈旧。

**结论与下一步**：A+B 的机制已在线实现——拒轮登记解开了训练行饥饿，`refit_every=1` 使每轮
都能重拟合且适配耗时仅约 1.5 s；但 60 s 量级的 fit 成本与"promotion 不等于实测改善"是新的
主瓶颈。下一步建议：① 以实测均匀度（而非内部 validation loss）作为 promotion 判据或加
早停，避免 r04 之后连续劣化；② 在确认判据前不再把 `refit_every=1` 当生产默认值。

## 8. beam quality 连通域索引复用与直接拟合（离线，实验待批）

提交 `0f32562896d56076cd6e3f192266ef15787d3730`；只改
`src/ao_shaping/gui/ccd/beam_quality/analyzer.py` 与
`tests/ao_shaping/gui/ccd/test_beam_quality.py`。

**算法改进**：

1. `_largest_component` 直接返回最大连通域的 `(ys, xs)`，`_ellipse_from_mask`
   与 `_box_centroid` 复用同一组坐标；去掉两处对全幅布尔图的重复 `np.nonzero`。
2. `fit_profile` 从 `curve_fit` 改为等价的有界 `least_squares`，保留解析雅可比、
   `max_nfev=4000`、失败返回 `None` 的口径。

**实验流程改进**：

1. 新增四步 OpenChamber 计划与临时交错计时脚本；每次变体都对六类输入逐字段对账：
   640×480、1344×760、平场、饱和、极小帧、多连通域。
2. 上下机门禁分开：本轮只做 CPU 离线验证，未占用 SLM #1、Daheng cam_id=0、激光或 GPU。

**离线证据**：

- 六类输入的 `FrameMetrics` 全部逐字段相同；`least_squares` 的拟合参数与 `curve_fit`
  相同。
- 15 轮 AB/BA 交错计时、每轮 15 次中位数：640×480 为 15.010 → 12.514 ms，
  节省 2.496 ms（16.6%，配对中位数 2.465 ms）；1344×760 为 17.347 → 13.550 ms，
  节省 3.797 ms（21.9%，配对中位数 4.095 ms）。绝对耗时随机器负载浮动，
  以同进程配对差值为准。
- `pytest tests/ao_shaping/gui/ccd -q --import-mode=importlib`：
  `102 passed, 1 skipped`；`ruff check` 与 `git diff --check` 均通过。

**未采用的候选**：mean/std 改 `np.einsum` 在代表帧上单独少 0.36–0.51 ms，但端到端
配对收益在 -1.042 ms 到 +0.777 ms 间摆动，16-bit 满量程的最大统计差为
0.00390625 counts；不引入该行为变化，也不新增对应容差测试。GPU 批量仍按 §2 不采用。

**实验申请与执行**：用户批准后运行
`.venv\Scripts\python.exe -m pytest tests/ao_shaping/gui/ccd/test_beam_quality_hardware.py -q --hardware -v --import-mode=importlib`。

1. 第一场 `21:39:49–21:41:56`：`1 failed in 131.58 s`，失败项为
   `visible["render_active"] is True`；闭环会话本身完成。根因是 `DesktopShellApp`
   恢复了上一场遗留的页 1 选中态，页 0 实际处于隐藏状态。测试中显式
   `shell.select_page(0)` 后消除该前置依赖。释放耗时 0.523 s，
   `beam_source=None`、`transfer_running=False`。
2. 第二场 `21:43:44–21:45:53`：`1 passed in 134.32 s`。页 1 会话完成，
   `journal.json` 与 `optical_probe.json` 均存在；页 0 可见段/隐藏段门禁通过。
   释放耗时 0.535 s，`beam_source=None`、`transfer_running=False`，
   `shell.close()` 正常关闭各页面。两场均未并行启动其他设备进程。

**设备占用 / 释放**：两场均占用 `SLM #1` + `Daheng cam_id=0`；页 0 用
`SimBeamSource`，不占相机。第二场证据
`data/debug/beam_quality/hidden_gate_hardware_20261127_optimize_v2.json`，
会话 `data/debug/slm_transfer/20261127_214406`。

**硬件阶段两栏结论**：算法改进为「无」——设备阶段不再改算法；实验流程改进是
硬件测试显式切回页 0，消除了对持久化工件当前页的依赖。下一步无需再次实验；
如果调整 promotion 判据或 mean/std 口径，再按新改动单独申请。
