# S1e 同字节探针判读报告（2026-11-26）

<!-- provenance:start -->
> **生成脚本**: [`scripts/slm/s1e_same_bytes_probe_judge_20261126.py`](../../../scripts/slm/s1e_same_bytes_probe_judge_20261126.py)
> **复现命令**: `python scripts/slm/s1e_same_bytes_probe_judge_20261126.py --out-dir report/shape_trials/s1e_same_bytes_probe_20261126`
> **数据/关联脚本**: [`scripts/slm/s1e_same_bytes_probe_capture_20261124.py`](../../../scripts/slm/s1e_same_bytes_probe_capture_20261124.py)
> **数据/关联脚本**: [`scripts/slm/issue247_axis_map_solve_20261121.py`](../../../scripts/slm/issue247_axis_map_solve_20261121.py)
> **运行环境**: 离线判读（只读 run_01 的 frames 与 capture_journal.jsonl；未占用 SLM/CCD/激光）
> **说明**: S1e 同字节探针（预注册 §12.16）采集后的离线判读：复用冻结纯函数 FROZEN_CRITERIA/axis_type/pairwise_max_distance/d6_zero_exception 与冻结求解器 find_spots/principal，按 §12.16.3 只执行不改口径，登记 D1–D6 与 §12.16.4 E 分支。判读不重定义任何阈值；漂移校正、谐波比、几何指纹仅作佐证，不参与判据。
<!-- provenance:end -->

## 摘要

- **目标**：把「同一 SLM 字节流是否确定给出同一远场」从 S1d 的偶发对照，升级为受控重复实验，分离「序列位置 / 前驱数组 / 会话上电态」三个候选变量（预注册 §12.16）。
- **采集**：变体 A（连续一次 `open`/`close`，12 上传 ×2 帧 = 24 帧），2026-11-26 05:07:21–05:07:50 UTC，`status=ok`，设备四标志全 `true`。
- **结论**：**D1 = `D-fail`**（H 六次上传主瓣两两最大 **2.431 px** > 1.5 px）。但 D-fail 的**性质是位置抖动**——H 六次上传**全部 `col-type`**、远场形状与主瓣距离恒定（93.3–95.2 px），仅**位置**在会话内漂移；**未复现 S1d 的轴型翻转**。按 §12.16.4 取**分支 E2（`delivery_nondeterministic_same_bytes`）**，并显式记录：该分支原预设「B 稳定 + 轴型可按前驱分组」在本轮**不成立**（B n=3 且同样抖动；H 臂轴型分组退化），故 E3 不适用。
- **状态**：S1 维持 **BLOCKED**；S1b/S1c/S1d 原始记录不改写；冻结测试集不动。

## 1. 预注册依据与不变量

- **权威来源 = git HEAD**：预注册 §12.16（采集计划）、§12.16.3（判据）、§12.16.4（分支）、§12.16.6/§12.16.7（机制更正与根因候选）。工作树中同名预注册文档若与本报告冲突，一律以 HEAD 为准。
- 采集脚本 blob `640d85c6`、离线测试 blob `80de251d`（均与 HEAD 一致，采集期间未改脚本）。
- **判据采前冻结、采后不改口径**：本报告只**执行**判据、不重定义任何阈值；漂移校正、谐波比、几何指纹均为**佐证量**，不参与判定。

## 2. 采集与数据 provenance

### 2.1 设备与参数

| 项 | 值 |
|---|---|
| SLM | #1，驱动板 `22030108` |
| 波长 | 1064 nm |
| 相机 | 大恒 MER2-507-23GM（`cam_id=0`），Mono12，`full_scale=4095` |
| 曝光 | **0.087 ms**（与 S1b/S1c/S1d 冻结一致，未改） |
| settle | 0.8 s/上传 |
| 量化 | `encoding_max_gray=1023` |
| 帧数 | 24（12 上传 ×2） |

### 2.2 时间线与释放

- `started_at_utc` 05:07:21.79 → `finished_at_utc` 05:07:50.30（UTC），`elapsed_s=28.50`。
- `released.json`：`slm_closed` / `camera_closed` / `lock_released` / `closed_normally` **全 `true`**，`released_at_utc=2026-11-26T05:07:50.296644+00:00`，`error=null`。
- 本判读为**离线**：只读 `run_01` 的 `frames/*.npy` 与 `capture_journal.jsonl`，**未占用 SLM/CCD/激光**，不采 dark、不操作激光。

### 2.3 数据清单与「同字节」前提

- 24 帧、`capture_journal.jsonl` 36 行；H 六次上传的**物理字节 SHA 全同**（`b98be9a6696c…`）、B 三次全同（`c8d937cedb29…`）——每臂内部「同字节」前提成立。
- 字节布局：H 数组 **F-contiguous**（strides `(2, 2400)`），B **C-contiguous**（`(3840, 2)`），flat C-contiguous。该差异见 §5.3。

## 3. 冻结判据（§12.16.3，原文摘要）

- **帧内一致性**：每上传 2 帧分别判读，主瓣位置差 ≤1.5 px 记为有效上传。
- **D1 同字节确定性**：H 六个有效上传主瓣两两最大欧氏距离 ≤1.5 px ⇒ `D-pass`，否则 `D-fail`；有效 <5 ⇒ `inconclusive`。
- **D2 轴型指纹**：`col-type` = `|dc|>1.5|dr|` 且 `dist∈[90,102]` px；`row-type` = `|dr|>1.5|dc|` 且 `dist∈[52,64]` px；其余 `other`。谐波比仅佐证。
- **D3 前驱规则**：若 D-fail，按 `flat/H/B` 前驱类型分组统计轴型；零例外可解释 ⇒ `history_dependent`，否则 `display_nondeterministic`。
- **D4 B 对照**：B 三次上传同样按 D1 判；弱信号（峰 < 0.10×flat 峰）不计数。
- **D5 arm P**：断电重启变体（未执行）。
- 斑点评判一律复用冻结求解器 `find_spots`/`principal`（`exclude_core=6`、`min_dist=15`、`thresh=0.01×max`）；0th 参考取**本变体** `flat` 质心 `(row=1105.829, col=1227.582)`，不跨变体复用。

## 4. 判读结果

### 4.1 逐上传读数

基准：`flat` 0th 质心 `(1105.83, 1227.58)`，flat 峰 2411；无 `weak` 上传（主瓣峰 548–628，均 > 0.10×2411≈241）。

| pattern_id | 数组 | 前驱 | 帧内差 px | dr | dc | dist px | 轴型 | 峰值 | 有效 |
|---|---|---|---|---|---|---|---|---|---|
| `flat` | flat | — | — | — | — | — | — | — | 参考 |
| `s1e_a1` | H | flat | 0.04 | −1.63 | −93.59 | 93.60 | col-type | 574 | 是 |
| `s1e_a2` | H | H | 0.87 | −1.85 | −94.19 | 94.20 | col-type | 614 | 是 |
| `s1e_b1` | B | H | 0.13 | −0.84 | −59.20 | 59.20 | other | 598 | 是 |
| `s1e_a3` | H | B | 0.66 | −2.13 | −93.32 | 93.34 | col-type | 586 | 是 |
| `s1e_b2` | B | H | 0.52 | −1.09 | −58.56 | 58.57 | other | 548 | 是 |
| `s1e_a4` | H | B | 0.32 | −1.96 | −94.31 | 94.33 | col-type | 623 | 是 |
| `flat_mid` | flat | — | — | — | — | — | — | — | 参考 |
| `s1e_a5` | H | flat_mid | 0.40 | −3.50 | −95.14 | 95.21 | col-type | 626 | 是 |
| `s1e_b3` | B | H | 0.64 | −3.45 | −58.90 | 59.00 | other | 596 | 是 |
| `s1e_a6` | H | B | 0.41 | −2.40 | −95.11 | 95.15 | col-type | 628 | 是 |
| `flat_post` | flat | — | — | — | — | — | — | — | 参考 |

图 1 三联图（`s1e_same_bytes_probe_20261126_fig1.png`）：(a) 主瓣偏移随上传序位；(b) flat 0th 质心会话漂移；(c) 轴型指纹 H/B 距离比。

### 4.2 D1 同字节确定性 → **`D-fail`**

- H 有效上传 n=6，主瓣两两最大欧氏距离 **2.431 px**（最大偏差对 `s1e_a1`–`s1e_a5`）> 1.5 px ⇒ **`D-fail`**。
- 性质：**位置抖动**，非轴型翻转（见 §5.1）。

### 4.3 D2 轴型指纹

- **H 六次全部 `col-type`**：dist 93.3–95.2 px（均值 **94.30**），`|dc|>1.5|dr|` 且落在 [90,102] 窗口。
- **B 三次全部 `other`**：dist 58.6–59.2 px（均值 **58.92**），`dc` 主导但 **58.9 ∉ [90,102] 且非 `dr` 主导**（不满足 `row-type` 的 [52,64] + `dr` 主导）。
- **H/B 距离比 = 1.6004 ≈ 面板宽高比 1920/1200 = 1.6000**（见 §5.2）。

### 4.4 D3 前驱规则 → **`display_nondeterministic`（但判别力=False）**

- 全量（H+B）前驱分组：`flat`→[col-type, col-type]、`H`→[col-type, other, other, other]、`B`→[col-type, col-type, col-type]；`zero_exception=False`。
- 仅 H 臂前驱分组：`flat`→[col-type, col-type]、`H`→[col-type]、`B`→[col-type, col-type, col-type]；`zero_exception_H=True`，且六次上传**全为同一轴型**（`degenerate=True`）。
- **判别力 `discriminating=False`**：D-fail 是**位置抖动**，而轴型分组在 H 臂内**退化为单一轴型**，无法据此区分历史效应；全量分组出现的「例外」仅因 **B 上传（本身即 `other`）跟随 H**，属**数组身份混杂**，非历史依赖。故 **E3 不适用**。

### 4.5 D4 B 对照 → **`inconclusive`**

- B 有效上传 **n=3 < 5** ⇒ 按冻结 D1 规则 `inconclusive`（不可判 `D-pass`/`D-fail`）。
- 佐证（不入判据）：B 主瓣也抖动，两两最大 **2.629 px**（`s1e_b1`–`s1e_b3`），**未显示比 H（2.431 px）更稳**；因此「排除全会话任意上传都不稳」这一步在本轮**未能完成**。

### 4.6 D6 槽位号分组

- 12 上传各占唯一 memory 槽（3–14），每组仅 1 上传 ⇒ `zero_exception_slot_groups`（**trivially**，1 上传/槽；不构成强证据）。

### 4.7 D5 arm P

- 变体 P（`run_02`，物理断电重启）**未授权、未执行** ⇒ `not_applicable`。

## 5. 关键分析

### 5.1 D-fail 的性质：位置抖动，而非轴型翻转

- **帧内**（同一上传 2 帧背靠背）位置差仅 0.04–0.87 px——单次上传内远场稳定。
- **跨上传**（秒级间隔）最大漂移 2.431 px（H）/ 2.629 px（B）。
- 0th 参考会话漂移仅 `flat_mid` 0.456 px、`flat_post` 1.103 px；**漂移校正后 H 仍 2.128 px、B 2.307 px > 1.5** ⇒ D-fail **非全局 0th 漂移所致**，属远场**相对 0th 的位置抖动**。
- 全程**无轴型翻转**：H 恒 `col-type` ~94.6、B 恒 `other` ~58.9，形状/距离恒定，仅位置变。

### 5.2 ×1.6 几何指纹跨会话复现

| 变量 | H（axis=1） | B（axis=0） | 比值 |
|---|---|---|---|
| S1e（本日） | 94.30 px | 58.92 px | **1.6004** |
| S1d（11-24） | 96.09 px（`grating_x_p128`） | 60.31 px（`grating_y_p128`） | 1.593 |
| 面板 | 1920 | 1200 | **1.6000** |

两日同向（`dc` 主导）、同量级、比值均 ≈ 面板宽高比 1.6 ⇒ 与 §12.16.6 登记的「旋转 90° + ×1.6」机制一致，且**跨两次会话稳定**，属**系统性（确定性）**效应，而非随机。

### 5.3 物理字节布局（F-contiguous）作为**系统性**根因候选

- H 数组为 F-contiguous，`driver.py:2660` 裸 `memmove` 使**物理字节序 ≠ 逻辑序**，LCOS 因而得到转置/置换图案（§12.16.7 根因候选）。
- **但**该布局在全部 H 上传中**恒定** ⇒ 它解释「为何 H 是 `col-type` 94.6、B 是 59」这一**常量指纹**，**不解释** 2.4 px 的**逐次**抖动（抖动需逐次变化的机制）。二者须**分开归因**：系统性几何 vs 逐次位置抖动。

### 5.4 谐波佐证

- H 三次/五次谐波比 2.98–3.06 / 4.99–5.07；B 3.02–3.08 / 4.99–5.07，均在预期带（≈3.0 / 5.0）内，支持主瓣为真实一阶衍射（谐波为高阶衍射），非伪峰。**仅佐证**。

## 6. 分支登记与结论

### 6.1 E 分支（§12.16.4）

| 分支 | 前提 | 本轮 |
|---|---|---|
| E1 `D-pass` | 同字节确定 | 不适用（D-fail） |
| E2 `delivery_nondeterministic_same_bytes` | D-fail + B 稳定 + 前驱无零例外 | **取用**（见限定） |
| E3 `history_dependent_delivery` | D-fail + 前驱零例外 | 不适用（前驱非零例外；H 轴型分组退化） |
| E4 `power_cycle_state_dependent` | 执行 arm P | 不适用（未执行） |

**登记：E2 `delivery_nondeterministic_same_bytes`。** 同字节未给出逐位相同远场（位置抖动达 2.43–2.63 px）。**限定与保留**：

1. 本轮非确定性表现为**位置抖动**，**非** S1d 的轴型翻转；
2. E2 原预设「B 稳定」**未成立**（B `inconclusive` 且同样抖动），「排除全会话任意上传都不稳」**未完成**；
3. 「前驱无零例外」仅因 **B 上传（本身 `other`）跟随 H** 而成立，属数组身份混杂，非真实历史依赖。

### 6.2 S1 状态与不变式

- **S1 维持 BLOCKED**；不做 D bracket；G1/G3/G4 继续暂缓。
- **冻结测试集不动**；**S1b/S1c/S1d 原始记录一律不改写**。
- 不据此冻结任何映射，不宣称闭环收益。

## 7. 任务项与后续

1. **[待用户授权]** 变体 P（`run_02`，物理断电重启）——E4 判定所需；未授权不得执行。
2. **位置抖动分辨**：加长 settle / 多次重复 / 0th 与目标同步跟踪，区分「投递抖动」与「宽瓣质心测量噪声」——建议作为下一受控变体（**非本轮口径**，不得回改本轮判据）。
3. **前驱扫描变体（E3 检验）**：须在 H 臂内制造**轴型差异**才具判别力；本轮退化为单一轴型。
4. **provenance 收尾**：全树 `sync_report_provenance.py --check` 与 `report/README.md` 索引重建**待持锁车道**完成——本工作树正被其它车道删除/改写 `issue247_*` 等报告，全局 sync 会误伤。

## 8. 指标参数表

| 指标 | 值 | 阈值/说明 |
|---|---|---|
| D1 H 最大两两距离 | **2.431 px** | ≤1.5 ⇒ pass；实测 fail |
| D1 H 有效上传 | 6 | ≥5 |
| 最大偏差对 | `s1e_a1`–`s1e_a5` | — |
| 帧内位置差（H） | 0.04–0.87 px | ≤1.5 |
| D2 H 距离（均值） | 94.30 px | col 窗口 [90,102] |
| D2 B 距离（均值） | 58.92 px | row 窗口 [52,64]，但为 dc 主导 ⇒ other |
| H/B 距离比 | **1.6004** | 面板比 1.6000 |
| 0th 漂移 flat_mid / flat_post | 0.456 / 1.103 px | 佐证 |
| 漂移校正后散布 H / B | **2.128 / 2.307 px** | 佐证（仍 >1.5） |
| D4 B | max 2.629 px，n=3 | <5 ⇒ inconclusive |
| 谐波比 H 3rd/5th | 2.98–3.06 / 4.99–5.07 | ≈3/5 佐证 |
| 谐波比 B 3rd/5th | 3.02–3.08 / 4.99–5.07 | ≈3/5 佐证 |
| 全局峰 / 饱和像素 | 2520 / 0 | — |

## 9. Provenance

- **判读脚本**：`scripts/slm/s1e_same_bytes_probe_judge_20261126.py`（复用冻结纯函数与冻结求解器；未重定义阈值）。
- **采集/测试**：`scripts/slm/s1e_same_bytes_probe_capture_20261124.py`（blob `640d85c6`）、`tests/.../test_s1e_same_bytes_probe_capture_20261124.py`（blob `80de251d`）。
- **数据**：`data/experiments/attribution_20261123/s1/s1e_same_bytes_probe_20261124/run_01/`（`data/*` 受 `.gitignore` 忽略，未入库）。
- **结论绑定 commit**：预注册/采集/测试在 HEAD `a564d3f2`；本报告与判读脚本的提交 id 见文末/提交记录。
