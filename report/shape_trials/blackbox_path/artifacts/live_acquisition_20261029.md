# 黑盒路径重放：同曝光物理采集结果（2026-10-29）

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`scripts/slm/replay_saved_phases_20261029.py`](../../../../scripts/slm/replay_saved_phases_20261029.py)
> **数据/关联脚本**: [`src/ao_shaping/tools/slm/transfer/replay_phases.py`](../../../../src/ao_shaping/tools/slm/transfer/replay_phases.py)
> **运行环境**: 实验（Santec #1、Daheng Mono12）同曝光同位深重放 + 离线核对
> **说明**: 把两轮已存相位在同一曝光与原生位深下重放，作为「36 倍曝光差来自整形相位而非光路」的可比性检验；记录逐帧饱和计数与设备释放状态。
<!-- provenance:end -->

把两次 2026-10-29 黑盒会话（`blackbox_path_20261029_194837` = r1、`blackbox_path_gs_20261029_201406` = r2）
归档的**同一条相位**重新下发，用**同一曝光**各采一遍，以此把「曝光/本底」与「相位质量」分开。
本次只做**采集**，不做拟合、不重跑流水线。

- 采集脚本：`scripts/slm/replay_saved_phases_20261029.py`
- 采集目录：`data/shape_trials/blackbox_replay_20261029_212830/`
- 精确命令：`$env:PYTHONPATH='src'; .venv/Scripts/python.exe scripts/slm/replay_saved_phases_20261029.py --execute`
- 设备互斥锁：`DeviceLock`（`%TEMP%\ao_shaping\device_mutex\bench.lock`），全程独占，结束时释放
- 结果：**成功**，`replay_live_20261029.status` = `exit_code=0`（38 次采集 / 114 帧）

## 相位出处（不被二次加工）

每条相位都取存档 `anchored/capture_XXXX.npz` 的 `model_phase`（128×128 控制网格，**已含载波**），
经同一条路径 `to_panel(model_phase) -> Santec.create_phase_from_array()` 下发，**不叠加载波、不重加校正**。
r2 的种子不是会话自己的 manifest 行（它是该会话 `model/<shape>_final` 那一帧），脚本用数组逐位相等校验过
（不相等即失败），因此两张会话的种子出处可查。`flat` 是控制网格全零（面板无载波）。

逐条 sha256 见 `config.json` 的 `phases[]`；原始相位未复制进本目录（`config.json` 记 `inputs` 与会话内文件名）。

## 曝光不是沿用历史 4.32 ms

历史 r2 会话在 4.32 ms 下拍摄，r1 会话是另一档；两档曝光不同，直接比较会把亮度差读成质量差。
本次按实测重新定档（`ladder.json` 57 行探针记录，`exposures.json` 为结论）：

| 组 | 覆盖相位 | 曝光（请求） | 设备回读 | 定档方式 |
|---|---|---|---|---|
| `common` | 全部 13 条（含 `flat`） | 0.132364 ms | 0.132 ms | 0.02 ms 探针 → 线性外推到 0.80×4095 → 回采校验 |
| `high` | 仅 r2 的 6 条 | 3.165133 ms | 3.165 ms | `common` 档下 r2 峰值过低 → 按同一目标单独定档 |

设备回读与请求值差 0.4 µs，是大恒把曝光量化到整数微秒所致（issue #32 的既有口径），不是设置失败。
上限判据取 0.90×4095；实采 114 帧的**全帧峰值为 2606/4095 = 63.6%**，**饱和像素 0**。
回读值逐帧记在 `manifest.json` 的 `exposure_ms_read`，同一条采集内 3 帧完全一致。

`common` 档由**全帧**峰值定档，而全帧最亮点是采样窗之外的零级；因此窗内信号远低于全帧峰值
（`flat` 窗内峰值仅 137）。这是定档口径的必然结果，见下文「读数口径」。

## 采集内容

| 项 | 值 |
|---|---|
| 采集次数 / 帧数 | 38 / 114（13 条 `common` × 2 轮 + 6 条 `high` × 2 轮，每条 3 帧） |
| 第 2 轮顺序 | 反序（抵消时间漂移） |
| 原始帧 | `raw/*.npy`，`uint16`，`(3, 1944, 2592)`，Mono12，memory 模式 `video_mode=0`，2π=993 灰度 |
| 本底（窗内中位） | `common` 15.3 counts；`high` 17.3–19.0 counts |
| 逐次记录 | `manifest.json`（峰值 / 掩膜指标 / 3 帧时间戳与曝光回读 / 写相位与读图耗时） |
| 设备状态 | `device_state.json`：`camera_connected=true`、`slm_open=true`、`bit_depth=12`、`video_mode=0` |

掩膜一律用 **r1 的规范网格**（`camera_xy` / `target` / `core`），因此 r2 的帧也落在与 r1 相同的采样窗与
目标框上，跨会话读数才可比。r2 自己的网格相对 r1 相差 1.0 px，仅在其自检对照里用到。

## 可复现性

同一相位两轮重采，框内峰值差：

- `common`：0.0%–5.7%（`flat` 5.7%，其余 13 条里 10 条 < 2%）
- `high`：0.1%–4.6%

即两轮之间只有百分之几的漂移，远小于下文 r1/r2 之间的差异。

## 同曝光读数（本底扣除后）

`ee_bg` = 目标框能量占采样窗总能量（逐帧减窗内中位并截断到 0 后计算）；`cv_bg` = 框内核区变异系数（同口径）；
`框内信号/ms` = 框内信号总计数 ÷ 曝光（counts·ms⁻¹，跨曝光可比）。

| 组 | 相位 | 框内峰值 | ee_bg | cv_bg | 框内信号/ms |
|---|---|---|---|---|---|
| common | flat | 137 | 0.069 | 3.89 | 108 |
| common | square_r1_seed | 1261 | 0.660 | 6.48 | 677 |
| common | square_r1_final | 1243 | 0.660 | 6.42 | 679 |
| common | triangle_r1_seed | 1199 | 0.608 | 3.64 | 624 |
| common | triangle_r1_final | 1138 | 0.608 | 3.51 | 624 |
| common | rectangle_r1_seed | 1193 | 0.685 | 8.63 | 704 |
| common | rectangle_r1_final | 1143 | 0.687 | 8.32 | 704 |
| common | square_r2_seed | 79 | 0.563 | 0.83 | 605 |
| common | square_r2_final | 69 | 0.555 | 0.76 | 590 |
| common | triangle_r2_seed | 135 | 0.502 | 0.75 | 513 |
| common | triangle_r2_final | 106 | 0.487 | 0.68 | 499 |
| common | rectangle_r2_seed | 55 | 0.604 | 0.97 | 598 |
| common | rectangle_r2_final | 42 | 0.588 | 0.81 | 584 |
| high | square_r2_seed | 1326 | 0.654 | 0.80 | 555 |
| high | square_r2_final | 1158 | 0.647 | 0.74 | 540 |
| high | triangle_r2_seed | 2444 | 0.583 | 0.73 | 468 |
| high | triangle_r2_final | 1888 | 0.564 | 0.66 | 455 |
| high | rectangle_r2_seed | 837 | 0.716 | 0.95 | 545 |
| high | rectangle_r2_final | 548 | 0.705 | 0.78 | 531 |

读法（同曝光 `common` 档为唯一可比口径）：

1. **进框能量差 10%–18%，不是数十倍。** r2/r1 的框内信号速率比：方 0.87–0.89、三角 0.80–0.82、矩形 0.83–0.85。
   历史比较里 r2 更暗，主要是 4.32 ms 与另一档曝光的差别，不能全算作相位质量。
2. **分布差别远大于总量差别。** r1 把光收成一个尖峰：框内峰值 1138–1261、`cv_bg` 3.5–8.6，
   而框区平均只比本底高 4.7–29 counts ⇒ 亮点占住框内很小一块，其余空着。
   r2 把光铺在框内：`cv_bg` 0.66–0.97，峰值/框区均值约 6，`high` 档强信号下结论不变（不是暗帧噪声造成的低 CV）。
3. **`ee_bg` 两侧同量级**（0.49–0.72），说明差异主要是**框内是否填平**，不是「光是否进得来」。
4. 两轮 3 帧的重复性 < 6%，故上述排序不是漂移。

## 读数口径与限制

- `common` 档为压住**全帧**峰值而定（全帧最亮点在采样窗外），窗内信号因此偏薄：
  r1 的框区均值只比本底高 4.7–29 counts，其 `cv_bg`/`ee_bg` 对中位本底的估计敏感。
  r1「尖峰、不是填充」的结论由**峰值**（1138–1261，远高于本底）支撑，不依赖薄信号的那一项。
- r2 的 `common` 帧峰值 42–135，仅 2.5–9 倍于本底；r2 的填充性结论以 `high` 档（峰值 548–2444）为准，
  两档给出同一结论（`cv_bg` 0.66–0.97 对 0.66–0.97）。
- 未做任何拟合或再优化；本节所有数都是采集帧的掩膜读数，不改动相位、不改动窗口。
- 逐行日志 `replay_live_20261029.log` 由 PowerShell 管道写入，**长行被控制台宽度截断在 ~100 列**；
  权威数值以 `config.json` / `ladder.json` / `exposures.json` / `manifest.json` 为准。

## 设备释放证据（`release.json`）

| 项 | 值 |
|---|---|
| `lock_released` | `true` |
| `lock_is_locked_after_release` | `false` |
| `lock_holder_after_release` | `null` |
| `lock_holder_before_release` | pid 451260，命令行为本脚本 `--execute` |
| `camera_connected_before_close` → `after` | `true` → `false` |
| `slm_open_before_close` → `after` | `true` → `false` |
| 时序 | 21:28:34 取锁 → 21:28:35 开相机 → 21:28:36 开 SLM → 21:31:11 采完并关 SLM → 21:31:12 释放锁 |

## 停止与落盘核对（本次收尾）

停止方式：采集按 `--passes 2` 自行走完，没有中断也没有强杀任何进程；`finally` 段按
关 SLM → 关相机 → 释放 `DeviceLock` 的顺序收尾。

| 核对项 | 结果 |
|---|---|
| 进程 | 启动器 pid 451260 已退出，无残留采集进程（另有两个 21:33:55 起的外部 pytest 进程，与本次采集无关，未处理） |
| 互斥锁 | 收尾后复核 `DeviceLock.is_locked()` = `False`、`holder()` = `None` |
| 设备 | `release.json` 记录相机 `true → false`、SLM `true → false` |
| 帧 | `raw/` 38 个 `.npy`，每个 `uint16 (3, 1944, 2592)` = **114 帧**；全帧峰值 2606/4095；**饱和像素 0**（逐文件复算，与 `manifest.json` 一致） |
| 曝光读回 | `common` 0.132 ms、`high` 3.165 ms，每次采集 3 帧读回一致，与逐帧时间戳一致 |
| 轮次 | `common` 13 条 / `high` 6 条各两轮，第二轮是第一轮的严格倒序（逐组复核） |
| 相位 | 13 条数组已离线抄入 `phases/` + `phases.json`，逐条 sha256 回读等于 `config.json` |
| 种子出处 | r1/r2 的 `anchored` 种子与同会话 `model` 终态逐字节相等（重新核对） |
| 孤儿文件 | `raw/` 无未登记文件，`manifest.json` 无缺失文件 |
| 部分数据 | 首轮作废目录只留 `ladder.json`（`raw/` 为空），保留未删 |
| Recorder | 无 `process.csv` / `process.h5` / 过程 GIF：本脚本只写原始 NPY + manifest JSON；同系列 r1/r2 会话目录也没有 Recorder 产物 |

两处已知缺陷（不影响上表任何数字）：

1. 日志末行 `采集完成: 38 次采集 / 38 帧` 把**采集次数**写成了帧数；实际帧数是 114
   （`manifest.json` 每行 `frames: 3`）。这是日志文案问题，数据未受影响。
2. 本文件放在 `artifacts/` 且未登记进 `scripts/_common/provenance.py::REPORTS`，所以
   `test_report_provenance.py` 的溯源块用例会指名它（同目录 `planning_20261029.md` 也
   未登记）。补登记 + `sync_report_provenance.py` 属提交前动作，本次不做。

## 附：第一轮（作废，无帧）

`replay_live_20261029_attempt1.log` / `.status`（`exit_code=1`）是同一次尝试的首次运行：
探针跑完、**未写任何原始帧**即崩在两处证据写盘（相机连接标志取到了方法对象；释放后的锁持有者
`LockHolder` 不是可 JSON 化标量）。设备已按上下文管理器正常关闭、锁已释放，随后修正
`device_state.json`/`release.json` 的序列化并重跑，即本文件所述的 `blackbox_replay_20261029_212830`。
失败那次的目录 `data/shape_trials/blackbox_replay_20261029_212302/` 只留 `ladder.json`，作为那次探针的痕迹。

未提交、未推送。
