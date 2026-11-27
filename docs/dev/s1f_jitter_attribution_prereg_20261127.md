# S1f 随机抖动归因探针预注册（2026-11-27）

<!-- provenance:start -->
> **设计源**: [`hw_window_20261123_attribution_prereg.md`](hw_window_20261123_attribution_prereg.md) §12.16（冻结口径，blob 不变）；S1e 修复后判读见 `report/shape_trials/s1e_postfix_20261126/`
> **现行计划**: [`next_experiment_plan_20261124.md`](next_experiment_plan_20261124.md) §1 / §2.P1
> **修复门禁**: [`test_santec_write_to_memory_contiguity.py`](../../tests/ao_shaping/drivers/slm/test_santec_write_to_memory_contiguity.py)
> **采集脚本（本轮新增）**: [`s1f_jitter_attribution_capture_20261127.py`](../../scripts/slm/s1f_jitter_attribution_capture_20261127.py)
> **判读脚本（本轮新增）**: [`s1f_jitter_attribution_judge_20261127.py`](../../scripts/slm/s1f_jitter_attribution_judge_20261127.py)
> **复用（冻结，不改）**: [`s1e_same_bytes_probe_capture_20261124.py`](../../scripts/slm/s1e_same_bytes_probe_capture_20261124.py)、[`s1e_same_bytes_probe_judge_20261126.py`](../../scripts/slm/s1e_same_bytes_probe_judge_20261126.py)、[`issue247_axis_map_solve_20261121.py`](../../scripts/slm/issue247_axis_map_solve_20261121.py)
> **运行环境**: 本文与所有脚本、门禁、计划模式清单均为**离线**产出，本轮**未占用 SLM/CCD/激光**；实验须经用户另行授权后按 §6 清单执行。
> **绑定 commit**: 本轮离线准备提交，见本文件末行（提交后回填短 id）。
<!-- provenance:end -->

## 摘要

- **目标**：S1e 修复后 D1 仍为 `inconclusive`、S1 维持 `BLOCKED`。残留的 ~1.6 px 随机抖动**无法由现有 S1e 判读区分**是相机读出/质心提取噪声、SLM 写屏随机性、还是光路慢漂移——因为 S1e 的 `intra_upload_dist` 是**同一次写屏、两帧之间**的质心差，写屏状态完全相同，该值只反映读取/提取层。本探针用**三个变体分层**在同一会话内分别量化三层贡献。
- **唯一变量**：S1e 只做「一次写屏采 2 帧 + 跨上传比较」；本探针改为三层结构（同写屏多帧 / 同字节重复写屏 / flat 头-中-尾），图样、相位 builder、曝光、判读求解器均复用冻结件，**不改冻结判据、不改 S1e 数据**。
- **本轮产出**（离线）：本预注册 + 采集脚本 + 判读脚本 + 双向离线门禁 + 离线准备报告。**不操作设备**。
- **状态影响**：本探针**不升格 S1、不改 S1e D1 口径、不做 D bracket、G1/G3/G4 继续暂缓、冻结测试集不动**。S1 解锁仍唯一依赖一个**另行预注册**的 D-pass。

## 1. 问题与归因框架

### 1.1 已知事实（S1e 修复后）

| 事实 | 数值 | 来源 |
|---|---|---|
| 修复后送屏字节（H，逻辑 C 序） | `7276d9da3b2a5ea536d56ea944adcfdcb38c98ad1dd01c1433d1b2219ac8218e` | `run_05_postfix` journal |
| 修复前送屏字节（H，F 序裸 buffer） | `b98be9a6…` | `run_02` journal |
| 轴型 | 修复前 `col-type` ~94 px → 修复后 `row-type` ~58 px | S1e 判读 |
| D1（同 array 多上传两两距离 ≤1.5 px） | 有效上传 4 < 5，pairwise 1.665 px → `inconclusive` | `s1e_postfix_official_20261127.json` |
| `intra_upload_dist`（同一次写屏两帧） | H `s1e_a2` 1.85、`s1e_a5` 1.68 px | `run_05_postfix` 帧记录 |
| flat 参考头尾漂移 | 1.225 px | S1e 判读 `drift_corrected_spread` |

**关键推断**：`intra_upload` 与跨上传 pairwise 同量级（1.68–1.85 vs 1.665 px），说明抖动大头**可能不在写屏随机性**，而在读取/提取层或漂移。但 S1e 无法把三者分开——`intra_upload` 里写屏状态固定，跨上传里写屏与读取混在一起。需要分层。

### 1.2 三层归因

| 层 | 探针变体 | 操作 | 度量 |
|---|---|---|---|
| L1 相机读出 / 质心提取噪声地板 | **变体 A**（同写屏多帧） | H 写屏**一次**，随后连采 24 帧，期间不重写 | 主瓣位置帧间散布 |
| L2 SLM 写屏随机性 | **变体 B**（同字节重写） | 同一 H 字节流重写 24 次，每次写屏后采 1 帧 | 主瓣位置写屏间散布 |
| L3 光路慢漂移 | **变体 C**（flat 头-中-尾） | flat 在会话头/中/尾各写屏一次，各采 16 帧 | 0 级中心随时间漂移 |

- **变体 A 是地板**：写屏状态全程不变，测得的是「给定同一远场，读取+提取能产生多大散布」。
- **变体 B 与变体 A 之差**（扣除地板后）即写屏随机性净贡献。
- **变体 C** 给出会话内漂移幅度与速率，用于判断漂移是否足以解释 ~1.6 px。
- 三者独立测量、同会话完成，避免跨会话不可比（memory 已记录：跨日 ms 不可比，只比步数）。

## 2. 冻结复用件（不改写）

| 件 | 用途 |
|---|---|
| `flat_phase` / `grating_phase` / `phase_to_gray`（`issue247_geometry_capture_20261120`） | 图样相位与灰度 builder，本探针不重实现相位数学 |
| `find_spots(frame, min_dist=20, thresh_frac=0.02)`、`principal(frame, ref, exclude_core=6.0, min_dist=15.0)`（`issue247_axis_map_solve_20261121`） | 亚像素质心与主瓣选取，与 S1e 判读同一求解器 |
| `frame_centroid` / `camera_identity` / `frame_filename` / `acquire_device_lock` / `utc_now` / `write_json` / `require_drive_board_entry`（`s1d_orientation_probe_capture_20261124`） | 设备锁、身份、文件名、时间戳等共享 helper |
| `FROZEN_PANEL_SHAPE=(1200,1920)`、`ENCODING_MAX_GRAY=1023`、`GRATING_PERIOD_PX=128`、`NATIVE_CCD_SHAPE=(1944,2592)` | 物理常数，来自 S1e |
| `run_05_postfix` 的 H 灰度字节 | 变体 A/B 的 H 数组须与该字节**逐字节一致**，采集脚本做断言 |

**不改写**：§12.16 冻结判据、S1b/S1c/S1d 与 A/P 原始记录、`s1e_same_bytes_probe_capture_postfix_20261126.py`、`s1e_same_bytes_probe_judge_20261126.py`、`driver.py` 本线守卫。

## 3. 探针契约

### 3.1 会话常数

| 参数 | 值 | 说明 |
|---|---|---|
| SLM | `#1`，`video_mode=0`，**全程一次 open / 一次 close** | 与 S1e 一致 |
| 相机 | Daheng `cam_id=0`，全幅 Mono12 | 与 S1e 一致 |
| 曝光 | **0.05 ms** | 沿用 S1e 修复后 `run_05_postfix` 的用户裁决曝光（G3 偏离已登记）；本探针不采 dark、不并行占用 |
| settle | 0.8 s（写屏后） | 与 S1e 一致 |
| 饱和门禁 | 帧内 `>= 0.9*4095` 计数 > 0 即中止该变体并报错 | 与 S1e 一致 |
| 设备锁 | `--wait-until-idle`，默认 1800 s | 见硬件纪律 |
| 帧总数 | 96（24 + 24 + 48） | 见 §3.2 |
| 激光 | **不操作** | — |

### 3.2 采集顺序（96 帧，单会话）

| 段 | pattern_id | array | 写屏次数 | 帧数 | 说明 |
|---|---|---|---|---|---|
| 1 | `flat_head` | flat | 1 | 16 | 变体 C 头部参考 |
| 2 | `s1f_a01`…`s1f_a24` | H | **1** | 24 | 变体 A：一次写屏后连采 |
| 3 | `flat_mid` | flat | 1 | 16 | 变体 C 中部参考 |
| 4 | `s1f_b01`…`s1f_b24` | H | **24** | 24 | 变体 B：每次写屏后采 1 帧 |
| 5 | `flat_tail` | flat | 1 | 16 | 变体 C 尾部参考 |

- 帧序号全局连续（0000 起），`frame_filename(index, pattern_id)` 命名，存于 `<out>/frames/`。
- 帧数合计：16 + 24 + 16 + 24 + 16 = **96**。
- 全程**一次** `slm.open()` / **一次** `slm.close()`；变体 A 的 24 帧在单次写屏后完成，变体 B 的 24 帧各对应一次写屏。

### 3.3 采集脚本接口

模块：`scripts/slm/s1f_jitter_attribution_capture_20261127.py`

- **计划模式（默认，无 `--execute`）**：只构建并打印确定性 `plan.json`；不建目录、不 import 驱动、不取设备锁。
- **执行模式（`--execute`）**：取锁 → 建 `out`（`exist_ok=False`）→ 写 `plan.json` + `capture_journal.jsonl` → 开设备 → 按 §3.2 采集 → finally 关相机/SLM、释放锁、写 `released.json` 与 `summary.json`。
- **CLI**：`--execute`、`--exposure-ms`（默认 `0.05`）、`--out`、`--wait-until-idle`（默认 `1800`）、`--slm-number`（默认 `1`）、`--cam-id`（默认 `0`）。
- **H 字节断言**：变体 A/B 送屏的 H 灰度数组 `tobytes(order="A")` 必须等于 `run_05_postfix` 的 `7276d9da…`（逻辑 C 序），不等则中止。
- **journal 事件**：
  - `upload`：`schema/event="upload"/write_index/arm/pattern_id/array/memory_number/phase_sha256/gray_sha256/physical_sha256/utc`
  - `frame_saved`：`schema/event="frame_saved"/index/pattern_id/arm/frame_in_arm/write_index/filename/frame_sha256/peak/median/saturated/saturated_count/centroid/utc/exposure_ms/bit_depth/full_scale`
- **arm 取值**：`flat_drift`（变体 C）、`same_write`（变体 A）、`repeat_write`（变体 B）。
- **released.json**：沿用 `issue247-s1d-orientation-probe-released/v1` 字段（`slm_closed`/`camera_closed`/`lock_released`/`closed_normally`/`error`/`elapsed_s`/`released_at_utc`）。
- **计划模式自检**须断言：段顺序、段内帧数、总帧数 96、H 字节断言逻辑、写屏次数与变体的对应。

### 3.4 判读脚本接口

模块：`scripts/slm/s1f_jitter_attribution_judge_20261127.py`

- **输入**：采集目录（`frames/*.npy` + `capture_journal.jsonl`）。
- **输出**：`<out>/s1f_jitter_attribution_judgment.json`（+ 打印表格）。
- **度量定义**：
  - `zero_order_reference(flat)`：`find_spots(flat, min_dist=20)` 最亮点的强度加权质心，取 `flat_head`/`flat_mid`/`flat_tail` 三组。
  - 变体 A 主瓣：对 `s1f_a01..a24` 每帧取 `principal(frame, ref=zr_head)` 的 `(row, col)`。
  - 变体 B 主瓣：对 `s1f_b01..b24` 每帧取 `principal(frame, ref=zr_mid)` 的 `(row, col)`。
  - 散布（每个变体）：`pairwise_max_px`（帧间两两 `hypot` 最大值）与 `rms_px = hypot(std_row, std_col)`（逐轴总体标准差）。
  - 漂移：`drift_head_to_tail_px = hypot(zr_tail - zr_head)`，另报逐轴差。
- **归因规则（本探针新预注册，非 S1e D 口径）**：
  - `readout_floor_rms = A.rms_px`
  - `write_screen_rms = max(0.0, B.rms_px - A.rms_px)`
  - `drift_px = drift_head_to_tail_px`
  - `ratio = B.rms_px / A.rms_px`（`A.rms_px <= 0` 时记 `ratio = None` + `ratio_note`，裁定 `undetermined`；不写 `inf`——共用 `write_json` 为 `allow_nan=False`，`inf` 无法落盘）
  - **裁定**：
    - `ratio <= 1.3` → **V1 读取/提取主导**：重写同一字节不显著增加散布，~1.6 px 抖动归因于相机读出 + 质心提取。
    - `ratio >= 2.0` → **V2 写屏随机性主导**：重复写同一字节显著移动远场。
    - `1.3 < ratio < 2.0` → **V3 混合**，报两分量占比。
    - `ratio is None` → **undetermined**（地板为 0，无法定级）。
  - **漂移标志**：`drift_head_to_tail_px >= 1.0` 记 `drift_material=true`，否则 false。
  - **交叉核对（仅佐证，非门禁）**：把 `A.rms_px` / `B.rms_px` 与 S1e `run_05_postfix` 的 `intra_upload_dist`（1.68–1.85 px）比较。

## 4. 离线门禁（本轮必须全过）

| 门禁 | 内容 |
|---|---|
| 计划模式只读 | 不建目录、不取锁、不 import 驱动 |
| 结构断言 | 段顺序、段内帧数、总帧数 96、写屏次数/变体映射、H 字节断言逻辑 |
| 判读纯函数 | 合成帧注入后，`zero_order_reference`/`principal`/散布/裁定分支均可离线复现 |
| 门禁命令 | 定向 pytest + `ruff` + BOM + 术语门禁（`term_unify_20261016.py` dry-run `0 个文件`） |
| 范围 | 只跑本线测试，不跑全量 pytest（本机 `scipy.fft(workers=-1)` 挂起已知问题） |

## 5. 设备纪律

- 实验前：离线脚本 + 预注册 + 门禁全部通过，向用户提交 §6 清单，**不直接操作设备**。
- 采集期间：**不判读**，只采集与记录；判读全部离线完成。
- 占用/释放：在回复中明确报告占用了什么、做了什么、何时释放、是否正常关闭（`released.json.closed_normally`）。
- 背对背 open/close 冷却 ≥ 200 s；带 `--wait-until-idle 1800`；不并行占用同一设备；不操作激光、不采 dark。

## 6. 实验清单（供用户授权后执行）

```powershell
# 1) 计划模式自检（只读，不取锁）
.venv\Scripts\python.exe scripts/slm/s1f_jitter_attribution_capture_20261127.py

# 2) 定向门禁
.venv\Scripts\python.exe -m pytest tests/ao_shaping/scripts/test_s1f_jitter_attribution_capture_20261127.py tests/ao_shaping/scripts/test_s1f_jitter_attribution_judge_20261127.py -q --import-mode=importlib

# 3) 上机采集（用户授权后，单会话 96 帧）
.venv\Scripts\python.exe scripts/slm/s1f_jitter_attribution_capture_20261127.py --execute --exposure-ms 0.05 --out data/experiments/attribution_20261123/s1/s1f_jitter_attribution_20261127/run_01 --wait-until-idle 1800

# 4) 离线判读（不占设备）
.venv\Scripts\python.exe scripts/slm/s1f_jitter_attribution_judge_20261127.py --capture-dir data/experiments/attribution_20261123/s1/s1f_jitter_attribution_20261127/run_01 --out report/shape_trials/s1f_jitter_attribution_20261127/
```

## 7. 产出与提交范围

- 本线文件：本预注册、采集脚本、判读脚本、两份离线测试、离线准备报告（`report/shape_trials/s1f_jitter_attribution_20261127/`）。
- **不碰** `scripts/_common/provenance.py`；`driver.py` 工作树中他车道 `SlotRotator` hunk 保持未暂存。
- 只提交本线文件，逐路径 `git add`。

## 8. 绑定 commit

- 离线准备提交：`aca0b987`（预注册、采集脚本、判读脚本、两份离线测试、离线准备报告）。

## 9. 修订记录：光强预检（2026-11-27，与归因阈值无关）

- **触发**：授权后 `run_01` 采集 96 帧 / 28 次写屏，设备正常释放，但全部帧无有效光强：
  flat 帧峰 41 counts（峰位 `(1,460)` 边缘噪声），同日同曝光 S1e `run_05_postfix` flat 峰 2524 counts（峰位 `(1105,1227)`）；
  帧均值 15.0 与 S1e 15.2 一致，写屏 digest 全部正确 → 相机读出链与 SLM 写屏均正常，缺的是光学输入。
  该会话的判读输出（A.rms ≈ 978 px、V1）系噪声峰跳变假象，**已作废**并移入
  `report/shape_trials/s1f_jitter_attribution_20261127/invalid_run01_no_light/`。
- **新增数据质量门禁**（不参与 §3.4 归因判据、不改阈值）：采集在第一帧写盘前检查
  `peak - median >= LIGHT_MIN_PEAK_CONTRAST`（默认 50 counts），否则中止、正常关设备并写 `summary.json.status=error`；
  判读在定位前用同一阈值拒绝无光会话（`JudgmentError`），避免输出假裁定。
- **复采**：须先恢复激光出光/快门，再按 §6 清单执行到 `run_02`；本探针本轮仍未改变 S1 状态。
