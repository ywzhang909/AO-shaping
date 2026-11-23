# Agent 当日记录 2026-11-23（占用设备：无。全程离线，未开 SLM / CCD / 激光）

隔离工作树 `.worktrees/dynamic-shaping-drift-loop`，分支 `feat/dynamic-shaping-drift-loop`。
**未 push、未改 issue 状态、未触碰 #234/#236 的任何证据或门控结论。** 本日是
"固定目标保持"（2026-11-22 §6）的后续：准备**物理实验用的测试代码**，软件仍只做完离线验证。

## 1. 标定链路上的一个真实 bug：窗口目标 ≠ 整帧 CCD

`_CalibrationThread` 原来把**分析窗口上的目标图**直接传给
`calibration_stats_from_frames`，而硬件 `CapturedFrame.frame` 是**整幅 CCD**。
两者形状不同，真设备采集必然在统计入口炸掉（离线测试用同尺寸假帧，所以没暴露）。

修复：`target_lock.py` 新增 `target_on_frame(xy, target, frame_shape)`，按 `xy`
的**绝对像素坐标**把窗口目标铺回整帧零场（窗口内是目标、窗口外是 0），暗场掩膜
因此落在窗口外背景上。`xy` 越界或网格形状不符直接 `ValueError`，不悄悄裁剪。
页面标定线程改为传入 `xy` 并先铺整帧再统计。

## 2. 实验脚本：`scripts/slm/target_lock_ondevice_20261123.py`

headless CLI，复用生产路径 `HardwareBench` → `ClosedLoopRunner(mode="hold")` →
`TargetLockSupervisor`，不引入 Qt。默认 **plan-only**（不 import 驱动、不建目录），
显式 `--execute` 才碰设备；`--wait-until-idle`（默认 300 s）走 `device_mutex`
并全程持锁，与 `dynamic_shaping_capture_20261120.py` 同口径。

两个阶段：

- `calibrate`：零相位 N 帧（默认 8）→ `CalibrationSample` → 固定倍数推导
  `LockConfig`。原始帧写 `calibration/calibration_frames.npz`，统计与推导阈值写
  `calibration/calibration.json`。**饱和、峰值未过暗场+10σ、暗场 σ=0 都直接判
  本次不可用（退出码 3），但证据照常落盘。**
- `hold`：全程目标保持，逐轮 `rounds.jsonl` + `events.jsonl` + `rounds.csv` +
  `summary.json`（首末质心、位置漂移 px、位置误差 max/RMS、评分首末/最优、
  监督器动作计数）。`--save-raw-every N` 可留原始帧。
  - 有 `--source-dataset` 时用 `OnlineTransferAdapter`（配合 `--adapt` 才会每轮
    warm-start 重拟合 + 晋级）；没有时用 `load_frozen_parameters` 构造
    `FrozenDefocusForward`，保证 hold 自检不缺物理预测模型。
  - 只跑 hold 时可 `--lock-config <calibration.json>` 复用已标定阈值。

漂移注入：`--drift tilt-x|tilt-y` 用 `DriftBench` 在每次 `apply_phase` 前叠一条
随写入次数线性张开的相位斜坡（`--drift-rad` 峰谷总幅、`--drift-start-write`、
`--drift-ramp-writes`）。这是**物理上真实的光斑移动**，且幅度/复现性全在软件内；
每次写入的注入量进 `trace`，"rad → CCD px" 由当天实测回答，不靠模型假设。

关键结构决定：`--stage both` 时全流程**只有一个 HardwareBench 实例、只打开一次
设备**——标定用裸 bench（零相位不掺漂移），保持在外面叠 `DriftBench`，再用
`PreopenedBench` 把 runner 的 `open()` 变成 no-op（`close()` 照常释放）。避免
背对背 open/close 触发驱动板假死。

退出码：`0` 完成；`2` 计划无效（含 plan-only 输入错误）；`3` 设备/帧质量/
保持 fault 让结论不可用。

## 3. 实验操作顺序（激光与设备已在线，由用户执行）

```powershell
cd D:\Projects\TIFO\AO-shaping\.worktrees\dynamic-shaping-drift-loop

# 0) 光路预检（六步探测，见 libs/skills/ao-optical-path-calibration/SKILL.md）
python scripts/slm/probe_optical_path_validity.py ...

# 1) 计划复核：不碰设备，确认中心/窗口/网格来自检查点
python scripts/slm/target_lock_ondevice_20261123.py `
  --out data/debug/target_lock/20261123_a `
  --checkpoint D:\Projects\TIFO\AO-shaping\data\debug\slm_transfer\20261104_192122

# 2) 标定：零相位 8 帧，先确认曝光不饱和（失败会写明原因）
python scripts/slm/target_lock_ondevice_20261123.py --execute --stage calibrate `
  --out data/debug/target_lock/20261123_a `
  --checkpoint D:\Projects\TIFO\AO-shaping\data\debug\slm_transfer\20261104_192122 `
  --exposure-ms 20

# 3) 保持 + 漂移注入：复用刚标定的阈值；--adapt 让前向模型随漂移重拟合
python scripts/slm/target_lock_ondevice_20261123.py --execute --stage hold `
  --out data/debug/target_lock/20261123_a `
  --checkpoint D:\Projects\TIFO\AO-shaping\data\debug\slm_transfer\20261104_192122 `
  --lock-config data/debug/target_lock/20261123_a/calibration/calibration.json `
  --source-dataset D:\Projects\TIFO\AO-shaping\data\debug\slm_transfer\20261104_192122\capture `
  --adapt --drift tilt-x --drift-rad 2.0 --max-rounds 40 --save-raw-every 5
```

脚本只产证据，**不代替物理验收**：是否"守住了固定像素质心"由操作员按
`summary.json` 的 `stats` 与原始帧判断。#247 轴映射未冻结，倾斜方向与 CCD 像素
方向的对应关系以当天实测为准。

## 4. 验证（全部离线）

- `ruff check`（脚本 + 测试 + target_lock + 页面）：无问题。
- host Python `AO_ALLOW_HOST_PYTEST=1`：`test_target_lock_ondevice.py`（16 项）+
  `test_target_lock.py` + `test_closed_loop_shaping.py` 共 **88 passed**。
- 主仓库 `.venv`（PYTHONPATH 指向本工作树）：`test_closed_loop_shaping_page.py`
  **20 passed**。
- plan-only 冒烟：真实检查点 `20261104_192122` 解析出 span=120（provenance）、
  origin=(1304.3, 942.6)、grid_size=128；未创建输出目录。
- BOM / 术语门禁：本次新增、修改的文件均无 BOM；`term_unify_20261016.py`
  dry-run 不再报告这些文件（仓库里另有 12 个历史文件的术语漂移，未在本轮范围）。
- 标定原始帧以**采集 dtype** 落盘（12-bit uint16 不被转成 float64）；
  `test_run_calibration_keeps_raw_frame_dtype_on_disk` 守住这条。

## 5. 待办

- 执行 §3 三步，把结论（噪声地板是否估得稳、推导阈值是否合用、缓变漂移下
  质心误差 max/RMS）回填本文件。
- 本轮改动已提交（本地，未推送）：待上机结论回填后再决定推送。
