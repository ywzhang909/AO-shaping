# Agent 当日记录 2026-11-27（beam quality 整形链路降本，离线）

本轮**未占用硬件**；上机验证清单见文末，须用户确认后再执行。
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

## 3. 上机清单（待用户确认，本轮未执行）

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

## 4. §5 数据缺口：采集/重建代码已离线备好（未上机）

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

## 5. §5 采集器上机清单（待用户确认，本轮未执行）

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

命令同 §5 上机清单、输出目录改为 `data/debug/closed_loop_timing_20261127_rerun2/`。
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
