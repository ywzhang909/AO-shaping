# 固定 CCD 目标保持 · 标定与保持物理实验 — 2026-11-23

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`scripts/slm/target_lock_ondevice_20261123.py`](../../scripts/slm/target_lock_ondevice_20261123.py)
> **数据/关联脚本**: [`scripts/slm/target_lock_probe_20261123.py`](../../scripts/slm/target_lock_probe_20261123.py)
> **运行环境**: 硬件
> **说明**: 固定 CCD 目标保持·标定与保持物理实验（2026-11-23，SLM #1 1064 nm + Daheng CCD MER2-507-23GM NIR，cam_id=0）：两次上机（0.10/0.05 ms）标定均成功、保持均未达成，归因于标定口径错位（整帧 vs 窗口）/适配种子 [WinError 2]/起点光斑不在窗口，非真实物理漂移；并含倾斜注入幅度扫（≈0.628 px/峰谷-rad）与 registration 等价性探针（整数平移逐点等价）。输入为 data/debug 磁盘证据，未据此完成物理验收。
<!-- provenance:end -->

## 摘要

本日按「固定 CCD 像素质心、单一最终目标形状（矩形 field 目标，半宽 10×10 px）的闭环目标保持」
路线在光路 A 上做了两次上机实验（`target_lock_20261123a` / `target_lock_20261123b`，SLM #1 1064 nm +
Daheng CCD `MER2-507-23GM NIR` SN `FJB24112232`，cam_id=0）。两次的**标定阶段都成功**（8 帧零相位、
无饱和、`total_counts≈1.55e6`），但**保持阶段都未达成**，**退出码 3（故障即停）**。本报告把
保持失败归因于**软件注入/口径问题，而不是真实物理漂移**，因此**不构成物理验收结论**；物理验收
留待按本次三个根因修复后的重跑（`target_lock_20261123c`，计划中）。

| 运行 | 曝光 | 标定 | 保持 | 失败归因 |
|------|------|------|------|----------|
| a（`20261123a`） | 0.10 ms | 成功 | 未开始 | 起始帧饱和（峰值 255/255、44 px 触顶），门控拦截 |
| b（`20261123b`） | 0.05 ms | 成功 | rounds=1 后冻结 | ① 标定口径错位（整帧 vs 窗口）→ `low_signal`；② 适配种子 `[WinError 2]`；③ 起点光斑不在窗口（偏移 177.9 px） |

三个根因均已离线数值实锤并修复（见 §4–§6）；同时补做了**倾斜注入幅度扫**（§3.3）与
**registration 等价性探针**（§3.2），用于确定下一次重跑的注入幅值（`--drift-rad 30`）。

> **验收结论：保持未验收。** 本次证据只证明「软件注入漂移与窗口口径不匹配导致保持无法开始」，
> 不证明「闭环无法保持真实漂移」。run b 实际注入仅 2.0 rad 峰谷（≈1.26 px，§3.3），
> **小于** 5 px 位置容差——即使保持成功也不能演示拒漂移；重跑必须加大注入。

## 1. 设备占用记录（硬件纪律）

| 阶段 | 设备 | 操作 | 结果 |
|------|------|------|------|
| run a | SLM #1 + Daheng CCD | `--execute --stage both`（一次开设备） | 标定后 hold 拦截，**退出码 3** |
| run a 结束 | SLM #1 | 回滚零相位、置 gray-0 | 已关闭（六步光路探针 `ok`，退出码 0） |
| run b | SLM #1 + Daheng CCD | `--execute --stage both`（一次开设备） | 标定后 hold 冻结，**退出码 3** |
| run b 结束 | SLM #1 | 回滚零相位、置 gray-0 | 已关闭 |

设备锁：plan-only 与执行均带 `--wait-until-idle 300`；run b `device_lock.status=idle`、
`waited_s=33.363`。两次运行之间无并行硬件进程；每次占用开始/结束均报告设备、操作与关闭状态。

## 2. 运行命令（可复现）

run b（`target_lock_20261123b`）的实测执行命令（与 `plan.json` 一致，`execute=true`）：

```powershell
.\.venv\Scripts\python.exe scripts/slm/target_lock_ondevice_20261123.py `
  --execute --stage both --out data/debug/target_lock_20261123b `
  --checkpoint data/debug/slm_transfer/20261104_192122 `
  --source-dataset data/debug/slm_transfer/20261104_192122/capture --adapt `
  --exposure-ms 0.05 --wait-until-idle 300 --drift tilt-x --drift-rad 2.0
```

run a 同上，仅 `--out .../target_lock_20261123a`、`--exposure-ms 0.1`。
`--checkpoint` 冻结模型 `20261104_192122`（`frozen_parameters.json`，grid 128），
`--centre auto`（缺省 = 检查点 origin `(1304.28, 942.63)`），窗口 span 120（128×128 网格、步 2）。

## 3. 证据与指标

### 3.1 run a（0.10 ms）

- 标定成功：8 帧、无饱和；hold 起始帧**饱和拦截**——`峰值 255/255、饱和 44 px、曝光 0.10 ms`，
  `state=fault`，`reason="起点相位未取得有效帧，未开始保持：…饱和…"`。
- 归因：0.10 ms 曝光下焦点过强。**改用 0.05 ms** 重跑（run b）。

### 3.2 run b（0.05 ms）

- **标定成功**：8 帧、`elapsed_s=5.652`、`total_counts=1,545,843`、`peak_counts∈[172,185]`、
  `saturated_fraction=0.0`、`dark_mean=0.307`、`dark_sigma=0.993`、`pixel_count=5,038,848`。
- **hold 冻结**：`elapsed_s=1.156`、`rounds=1`、`state=fault`、
  `reason="监督器状态 converging：监督器未给出决策（已冻结或帧未通过门禁），帧原因：low_signal；尚无有效评分"`、
  `adapter="适配未就绪：开场标定不可用：[WinError 2] 系统找不到指定的文件。"`。
- round 0（帧 9）：`gate_ok=true`、`gate_reasons=[]`、`peak_counts=182`、`saturated_fraction=0.0`、
  `centroid_xy=(1299.619, 963.709)`、`metrics=null`（基线帧，尚未计分）。
- **帧原因**：hold 门禁的 `min_total_counts=30,916.86` 来自**整帧** 2% 口径
  （`0.02×1,545,843=30,916.86`），而监督器实际测的是**窗口网格帧**总光量 ≈22,432
  （`0.02×22,432.1=448.77`）→ 窗口帧总光量 < 阈值 → `low_signal` → 监督器冻结。

### 3.3 倾斜注入幅度扫（`target_lock_probe_20261123`）

`FrozenDefocusForward` 上逐点注入倾斜相位（1–80 rad 峰谷，每轴，含 wrap 对照）：

| 轴 | 斜率（px / 峰谷-rad） | 80 rad 处位移 | wrap_diff（归一化相对差） |
|----|----------------------|---------------|--------------------------|
| tilt-x → dy | 0.6278 | −50.23 px | ≤1.16e-7 |
| tilt-y → dx | 0.6246 | +49.97 px | ≤1.16e-7 |

交叉耦合矩阵 `J ≈ [[0.017, 0.625], [−0.628, 0.011]]`（`∂(dx,dy)/∂(tilt-x,tilt-y)`）；
1–80 rad **严格线性**，wrap 无伪影。结论：倾斜是**有效的位置执行器**（≈0.628 px/rad，对角强、
交叉弱），闭环反馈可用 `tilt = −g·J⁻¹·e` 把质心拉回基线。

**注入幅值定论**：run b 实际注入 `2.0 rad` 峰谷 → ≈1.26 px，**小于** 5 px 位置容差，无法演示拒漂移。
依扫定结果，重跑取 **`--drift-rad 30`** → ≈18.8 px（`0.628×30`），超过 5 px 容差，且相对
120 px 窗口（半窗 60 px）留有余量。`wrap_diff` 对照证实该幅值仍在 wrap 线性区。

### 3.4 registration 等价性（探针 ②）

- 光斑质心（8 帧零相位实测）`(1227.66, 1103.16)`，`σ=0.50 px`；冻结 origin `(1304.28, 942.63)`；
  偏移 `(dx,dy)=(−76.61, +160.53)`、`|offset|=177.87 px`。
- **整数平移** `(dx,dy)=(−76, +160)`（`round(spot)−round(origin)`）+ registration
  `(dy,dx)=(+160,−76)`：与 origin 窗口+无 registration **逐点等价**（`norm_relative_error=0.0`）。
- **分数平移** `(−76.61, +160.53)`：`norm_relative_error=0.113`——`FrozenDefocusForward` 无
  分数 registration，**不可用**（会引入 ~11% 形变）。故取整数平移。
- 前向单次构造：首次 8.964 s（含 CUDA 初始化）/ 热身后 0.442 s——逐轮 rebuild 可行（量级参考）。

### 3.5 阈值与窗口（探针 ④，修复后口径）

窗口网格帧（14641 px、步 2）：`dark_mean=0.692`、`dark_sigma=0.799`、`peak≈184`、
`total≈22,432`；修复后阈值 `min_total_counts=448.77`（=2%×22,432.1）、`min_peak_counts=16.68`、
`saturation_counts=249.9`；8 帧标定帧**全部通过**（`all_frames_pass=true`，`total∈[22210,22631]`）。
—— 修复后窗口口径与 hold 门禁一致（对照修复前 `30,916.86` 的错位）。

## 4. 根因 ①：标定口径错位（整帧计数 vs 窗口图计数）→ 已修

- **证据**：标定 `calibration_stats_from_frames` 用**整幅 CCD 帧**（5,038,848 px）求 2% 阈值，
  得 `min_total_counts=30,916.86`；hold 监督器对**窗口网格帧**（14,641 px）求总光量 ≈22,432，
  远低于阈值 → 首帧后即 `low_signal` 冻结（rounds=1）。
- **修复**（Lane C，`closed_loop_shaping_page.py`）：标定线程改按窗口口径——先
  `sample_on_grid(item.frame, self._xy)` 取窗口网格帧，再
  `calibration_stats_from_frames(window_frames, target=self._target, …)`。目标本是窗口图，
  与窗口帧同形，直接使用。修复后 `min_total_counts=448.77`，8 帧全部通过（§3.5）。
- 附带序列化修复：`on_frame` 用 `gate.saturation_pixels` / 帧元素数计算真实
  `saturated_fraction`；输入缺失写 `null` 不伪造 `0`（5 个回归测试）。

## 5. 根因 ②：适配种子 `[WinError 2]`（manifest 载荷缺失）→ 已去 `--adapt`

- **证据**：`--adapt` + `--source-dataset .../20261104_192122/capture` 时适配器构造失败，
  `adapter="适配未就绪：开场标定不可用：[WinError 2]"`。`--adapt` 是**可选**模型细化
  （transfer 在线适配，默认关闭）；它不改变闭环保持的物理测试结论。
- **修复**：重跑去掉 `--adapt`（离线标定已含 frozen 模型，`DriftBench` 用
  `FrozenDefocusForward` 即可）。标定建议须可见、可复核；输入变化后不再应用过期建议。

## 6. 根因 ③：起点光斑不在 origin 窗口 → 已修（auto-spot + 整数 registration）

- **证据**：`--centre auto` 用检查点 origin `(1304.28, 942.63)` 开窗，但实测光斑在
  `(1227.66, 1103.16)`，偏移 177.87 px（≈1.5 个窗口半宽）→ 窗口基本不含光斑，标定/保持
  口径失真。
- **修复**（Lane B）：`--centre auto-spot`——标定前采一帧零相位图定位光斑，窗口开在
  `round(质心)=(1228,1103)`，并把同一整数平移写进物理前向的 registration
  （`registered_forward_params` 用 `dataclasses.replace(params, registration_shift_ccd_px=(dy,dx))`，
  `(dx,dy)=(−76,+160)`）。探针 ② 证实整数平移逐点等价、分数平移不可用（§3.4）。
- 标定与保持**同曝光**（0.05 ms），保持模式强制 `candidate_policy="auto"`。

## 7. 车道代码变更（本分支未推送，含修复）

| 车道 | 内容 | 涉及文件 |
|------|------|----------|
| A | 闭环漂移反馈：`_drift_feedback` 用 `tilt=−g·J⁻¹·e`（`J` 按相位形状缓存，奇异判据 `cond>1e6` 跳过并告警）；反馈只进下发指令、不并入 `summary.applied_phase`（记纯整形相位）；默认 `drift_feedback=False`、gain 0.5、上限 10 rad | `closed_loop_shaping.py` + `test_drift_feedback.py` |
| B | auto-spot + 整数 registration + 探针幅度扫 + 6 新测试 | `target_lock_ondevice_20261123.py`、`target_lock_probe_20261123.py`、`test_target_lock_ondevice.py` |
| C | 标定改窗口网格口径 + GUI 标定线程同步（失败信号回页面） | `closed_loop_shaping_page.py`、`test_closed_loop_shaping_page.py` |

三车道合并后目标套件 **126 passed**（`test_target_lock`、`test_target_lock_ondevice`、
`test_closed_loop_shaping`、`test_closed_loop_shaping_page`、`test_drift_feedback`，
`--import-mode=importlib`）。

## 8. 门禁与代码纪律

- 3 处 `except Exception: pass`（runner 收尾回滚、runner 收尾关闭、页面标定线程关闭）
  改为 `logger.warning(...)`（`{}` 占位格式），不再静默吞错；1 处未注明判据的盲捕获补
  `# 标定失败必须回到页面…` 判据注释。
- 盲捕获/静默计数随之为**实测值**（合并 master 后统一设定，见 §9 说明与
  `tests/ao_shaping/test_conventions.py` 常量旁注释 + `TODO.md` §8.9 理由）。
- 报告溯源按 `docs/dev/report_provenance.md` 三处同步（`provenance.py::REPORTS` 加键、
  `sync_report_provenance.py` 生成溯源块与 `report/README.md` 索引、`test_report_provenance.py` 校验）。

## 9. 结论与下一步

- **本次保持未验收**：两次均为「软件注入/口径问题导致保持无法开始」，不是真实物理漂移；
  run b 注入 2.0 rad（≈1.26 px）本就小于 5 px 容差，成功也不能演示拒漂移。
- **修复已就位**：窗口口径标定（①）、去 `--adapt`（②）、auto-spot + 整数 registration（③）、
  注入幅值定为 `--drift-rad 30`（≈18.8 px）。
- **下一步（可选上机重跑）**：`target_lock_20261123c`，0.05 ms、无 `--adapt`、
  `--centre auto-spot`、`--drift-rad 30`；成功后回填本报告的保持结论（物理验收）。
- 设备均关闭（SLM #1 gray-0）；未推送，等待并入主分支。
