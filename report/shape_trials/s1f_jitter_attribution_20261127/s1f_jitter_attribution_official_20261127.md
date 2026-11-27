# S1f 随机抖动归因正式判读报告（2026-11-27）

<!-- provenance:start -->
> **生成脚本**: [`scripts/slm/s1f_jitter_attribution_judge_20261127.py`](../../../scripts/slm/s1f_jitter_attribution_judge_20261127.py)
> **复现命令**: `python scripts/slm/s1f_jitter_attribution_judge_20261127.py --capture-dir data/experiments/attribution_20261123/s1/s1f_jitter_attribution_20261127/run_02 --out report/shape_trials/s1f_jitter_attribution_20261127`
> **数据/关联脚本**: [`scripts/slm/s1f_jitter_attribution_capture_20261127.py`](../../../scripts/slm/s1f_jitter_attribution_capture_20261127.py)
> **运行环境**: 硬件采集 run_02（SLM #1 + Daheng cam0）+ 离线判读；未操作激光
> **说明**: S1f 随机抖动归因正式判读报告（2026-11-27）。预注册 docs/dev/s1f_jitter_attribution_prereg_20261127.md（含 §9 光强预检修订），设计源 docs/dev/hw_window_20261123_attribution_prereg.md §12.16。run_01 全场无光已作废（证据见 invalid_run01_no_light/）；run_02 有效：ratio=B/A=1.165 ≤ 1.3 → V1读取/提取主导（85.8%），write_screen_rms=0.136 px，drift=0.254 px 不材料。前序 commit aca0b987 / 540aeb83；判读产物与报告 cb2ac1da，回填 33639637。S1 维持 BLOCKED。
<!-- provenance:end -->

## 摘要

- **目标**：为约 **1.6 px** 的随机抖动另开探针，用三变体分层实验单独归因——分离「读取/提取地板」与「写屏随机性」，判定 ~1.6 px 抖动的主导来源。
- **方法**：同会话单次实验，96 帧 / 28 次写屏，三段：变体 A（同写屏连采 24 帧，测读取/提取地板）、变体 B（同字节 H 重复写屏 24 次各 1 帧，测写屏随机性）、flat 头/中/尾各 16 帧（测漂移）。曝光 0.05 ms 沿用 S1e 用户裁决。
- **采集结论**：`run_01` 全场无光（flat 峰 41 counts），**作废**；恢复光路后复采 **`run_02` 有效**（首帧对比度 1826、全帧峰 876–2101、0 饱和、写屏摘要唯一值正确）。
- **判读结论**：**裁定 V1 —— 读取/提取主导**。`readout_floor_rms = 0.820 px`，`write_screen_rms = 0.136 px`，`ratio = 1.165 ≤ 1.3`。即该 ~1.6 px 抖动的主导分量来自相机读取/提取链，写屏随机性仅占 14.2%，且全程漂移不显著。
- **状态影响**：本探针**不改变 S1 状态**；S1 仍 `BLOCKED`，其解锁路径仍取决于 S1e D1 裁定。

## 1. 任务项

| 项 | 内容 | 状态 |
|---|---|---|
| 离线准备 | 采集/判读脚本、两份测试、预注册、门禁自检 | 完成（`aca0b987`） |
| 实验采集 `run_01` | 96 帧 / 28 写屏，设备正常释放 | 完成，**数据作废（无光）** |
| 光强预检门禁 | 采集第一帧验光 + 判读无光拒判 | 完成（`540aeb83`） |
| 实验采集 `run_02` | 96 帧 / 28 写屏，设备正常释放 | **完成，数据有效** |
| 离线判读 | 按冻结 §3.4 口径出裁定 | **完成** |
| 正式判读报告 | 本文件 + 判读产物 | **完成** |

## 2. 采集

### 2.1 `run_01`：无光作废

`run_01` 于 2026-11-27T05:09Z 采集，96 帧 / 28 次写屏、设备正常释放，但**全程无有效光强**：
flat 帧峰仅 41 counts（峰位 `(1,460)` 图像边缘噪声），而同日同曝光的 S1e `run_05_postfix` flat 峰为 2524 counts；帧均值 15.0 与 S1e 15.2 一致、写屏摘要全部正确，故相机读出链与 SLM 写屏均正常，**缺的是光学输入**。该次判读输出（A.rms ≈ 978 px、V1）系噪声峰跳变假象，已作废，证据存于 `invalid_run01_no_light/`，不得引用。失效根因与处置详见 [`CAPTURE_INVALID_evidence.md`](invalid_run01_no_light/CAPTURE_INVALID_evidence.md)。

### 2.2 `run_02`：有效采集

设备占用 **SLM #1 + Daheng cam0**，采集完成后 `released.json` 四项标志全 `true`、`error=null`，`released_at_utc = 2026-11-27T05:32:10Z`，设备正常释放。全程**未操作激光、未采 dark**，采集期间不判读。

| 指标参数 | 值 |
|---|---|
| 采集目录 | `.../s1f_jitter_attribution_20261127/run_02` |
| `status` | `ok` |
| 帧数 / 写屏次数 | 96 / 28 |
| 曝光 | 0.05 ms |
| 位深 / full scale | 12-bit / 4095 |
| 光强预检（§9） | 首帧 peak 1841 / median 15.0 / 对比度 1826 ≥ 50，`passed=true` |
| 帧峰范围 | 876 – 2101 counts |
| 饱和帧数 | 0 |
| 写屏摘要 | H 唯一 `7276d9da…`；flat 唯一 `cb228af4…`（与 S1e `run_05_postfix` 一致） |

采集段布局（写屏次数 / 帧数）：flat_head 1/16 → A 1/24 → flat_mid 1/16 → B 24/24 → flat_tail 1/16。

## 3. 判读结果

### 3.1 0 级参考（flat 逐帧 max-peak 质心，逐轴中位）

| 参考 | 来源 | 帧数 | row (px) | col (px) |
|---|---|---|---|---|
| zr_head | flat_head | 16 | 1105.281 | 1228.530 |
| zr_mid | flat_mid | 16 | 1105.012 | 1229.353 |
| zr_tail | flat_tail | 16 | 1105.039 | 1228.452 |

### 3.2 抖动变体

| 变体 | 含义 | 参考 | 帧数 | rms_px | pairwise_max_px | std_row | std_col |
|---|---|---|---|---|---|---|---|
| A | 同写屏多帧（读取/提取地板） | zr_head | 24 | **0.820** | 2.254 | 0.555 | 0.603 |
| B | 同字节重写（写屏随机性） | zr_mid | 24 | **0.955** | 2.868 | 0.744 | 0.600 |

### 3.3 漂移（flat 头 → 尾）

| 量 | 值 |
|---|---|
| head_to_tail_px | 0.254 |
| dr (row) | −0.242 |
| dc (col) | −0.078 |
| `drift_material`（≥ 1.0 px） | **False** |

## 4. 归因裁定

| 量 | 值 |
|---|---|
| `readout_floor_rms` | 0.820 px |
| `write_screen_rms` | 0.136 px |
| `drift_px` | 0.254 px |
| `ratio = B.rms_px / A.rms_px` | **1.165** |
| 分量占比 | 读取/提取 **85.8%** / 写屏 14.2% |
| 裁定 | **V1 —— 读取/提取主导（readout/extraction dominant）** |

裁定依据冻结 §3.4 阈值：`ratio 1.165 ≤ RATIO_V1_MAX (1.3)`，落在 V1 区间；`A.rms_px > 0`，无 `ratio_note`、不触发 `undetermined`。

**解读**：该 ~1.6 px 随机抖动的主导来源是**相机读取/提取链**，而非 SLM 写屏随机性。写屏重复同字节 H 带来的额外抖动仅 0.136 px，比读取地板低约 6 倍；全程 drift 0.254 px < 1.0 px，不构成材料性漂移。这一结论意味着：若后续要压缩该抖动，有效方向是读取/提取侧（采集/质心提取），而非继续在写屏字节上做确定性重试。

## 5. 交叉核对（仅佐证，非门禁）

| 量 | 值 |
|---|---|
| S1e `run_05_postfix` intra_upload_dist | 1.68 – 1.85 px |
| A.rms_px（本次） | 0.820 px |
| B.rms_px（本次） | 0.955 px |

> 仅作佐证：S1e 同写屏两帧质心差为 1.68–1.85 px，与本次 A/B 的 rms 量级在同一范围，口径相互一致，但本探针的归因判据不依赖该佐证。

## 6. 数据质量门禁（预注册 §9，非归因判据）

- 采集在第一帧写盘前检查 `peak − median ≥ LIGHT_MIN_PEAK_CONTRAST (50)`，不达标即中止并正常关设备（`summary.json.status=error`）。
- 判读在定位前用同一阈值拒绝无光会话（`JudgmentError`），不再输出假裁定。
- 38 项定向测试、ruff、术语门禁均通过；`run_01` 实测判读已确认退出码 2 且不写输出。

## 7. 后续任务项

| 任务 | 说明 |
|---|---|
| 读取/提取侧优化 | 本裁定指向读取链，后续若压抖动应在此侧立项（采集读数或质心提取口径），写屏确定性方向预期收益有限 |
| S1 状态 | **维持 `BLOCKED`**，本探针不改变 S1 状态；S1 解锁仍取决于 S1e D1 裁定 |
| 复现 | 按 §「复现命令」重跑判读即可复核本报告全部数值 |
