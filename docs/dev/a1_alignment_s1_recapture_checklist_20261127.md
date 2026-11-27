# A1 光路对齐 + S1 六帧重采实验清单（2026-11-27）

> 状态：**离线备妥，等待用户授权实验**。本文不含任何已执行的设备操作；
> 按 `docs/dev/hardware_discipline.md` 与 `libs/skills/ao-optical-path-calibration/SKILL.md`
> 逐项执行，占用/释放须在回复中明确报告。

---

## 1. 为什么必须先对齐（证据与边界）

§12.11.5 六帧采集（`scripts/slm/capture_s1_mapping_disambiguation_20261124.py`）的冻结判据①逐字：

> flat_scale993 复现 A1 0 级参考 (row 1105.249, col 1227.556)，偏移超过 5 px
> 即判本次会话与 A1 不同束流，六帧全部作废。

- 冻结参考：`ZERO_ORDER_REFERENCE_PX = {row: 1105.249, col: 1227.556}`，容差 5 px
  （A1 实测轨迹见 `report/shape_trials/s1e_same_bytes_probe_20261126/s1e_same_bytes_probe_20261126.json`
  的 `supplementary.flat_reference_track`；`docs/dev/agent_notes_20261121.md` §2 亦记同一坐标）。
- 11-26 首次重采（`data/debug/slm_transfer/s1_verify_20261126_retry/summary.json`）：
  flat_scale993 质心 **(1018.9, 1273.0)**，相对参考偏移 **[-86.4, +45.5] px**
  （欧氏距离 ≈97.7 px），远超 5 px ⇒ **判据①证伪，六帧按冻结文本作废**。
- ~~归因：本次 Session 与 A1 是**不同束流**（光路状态不同）~~ —— **该归因已被 §6.5 推翻
  （2026-11-27）**：脚本的 `_centroid_px` 被全帧背景噪声主导（噪声权重占 61%），
  输出的 97.7 px 是光斑与噪声的加权混合（残差 0.0000 px），同一帧任何带掩模的估计都把
  0 级放在 A1 参考 0.90–1.38 px 内。故**不是不同束流、不是光路未对齐**，是测量函数的假阴性。
  脚本的采集/锁/释放/证据链本身正常（`status=ok`、6 帧、0 饱和、`closed_normally=true`）。
- 因此物理调光路**无法**让判据①通过（需光斑权重约 48 倍），重采六帧只会得到同样的
  ~100 px 假偏移。下一步需用户裁定判据① 的测量口径（见 §6.5 与汇报选项），
  任何「软件补偿位移」「改用别的 0 级参考」「放宽 5 px」仍不允许。

## 2. 现状核对（2026-11-27，离线）

| 项 | 值 / 判据 |
|---|---|
| 台面锁 | `bench.lock` 互斥字节 = 0 ⇒ 空闲 |
| 设备进程 | 无 SLM/CCD 相关进程（仅 MCP fetch/time 服务） |
| SLM #1 / CCD cam 0 / 激光 | 当前均**未被占用**；实验前仍须现场复核 |
| SLM 冷却 | 上次释放约 `2026-11-26 21:55` 本地；须按实际退出时刻算 `210 − 已过秒数`，>0 才等待，够 200 s 可直开 |
| 代码改动 | 仅新增对齐探针与离线测试；**未改** S1 采集脚本 / 冻结判据 / 既有数据 |

## 3. 执行顺序（用户授权后，逐段执行）

### 3.0 前置门禁（缺一不采集）

1. 现场确认 SLM #1 驱动板 ID 可读（`22030108`）、大恒 cam 0 空闲、无他人占用台面。
2. 按 §2 计算 SLM 冷却剩余；按 `ao-optical-path-calibration` SKILL 完成六步光路探测，
   `verdict=ok` 才继续（记录 `optical_path_record.json`）。
3. 所有设备命令带 `--wait-until-idle 1800`；全场采集、不做物理开窗。

### 3.1 物理对齐（人工台面操作，先于任何采集脚本）

- 目标：flat 0 级光斑在 CCD 全幅 (1944, 2592) 上从 (1018.9, 1273.0) 移到
  **(1105.249, 1227.556)**，即行向 **+86.4 px**、列向 **−45.5 px**。
- 只做光路机械调整（反射镜/平移台），**不动** SLM 灰度 LUT、不加倾斜/离焦相位、
  不改相机 0.087 ms 曝光与 12-bit、不改激光器功率设定
  （§12.11.5 冻结 `focus_d_m1=0.0`、无 Fresnel 面板）；若探针帧饱和，
  只用中性衰减片/光路衰减把峰值压回量程内，并记录该改动。
- 每调一轮，用 §3.2 轻量探针测一次质心，收敛到 5 px 内为止；不要凭肉眼判断后直接采六帧。

### 3.2 轻量对齐探针（脚本离线备妥，尚未实验执行）

```bash
python scripts/slm/probe_a1_zero_order_alignment_20261127.py --self-test
python scripts/slm/probe_a1_zero_order_alignment_20261127.py \
  --out data/debug/slm_transfer/a1_align_probe_<序号>
# 确认 plan 无副作用后，才加 --execute（须全新目录）：
python scripts/slm/probe_a1_zero_order_alignment_20261127.py \
  --execute \
  --out data/debug/slm_transfer/a1_align_probe_01 \
  --wait-until-idle 1800
```

- 探针只显示 flat_scale993（scale 993 / modulus 994），逐帧算质心（复用 S1 的
  `_centroid_px` 口径），写 `alignment_measurement.json`（逐帧与均值质心、drow/dcol/dist、
  `within_tolerance`、`any_frame_saturated`）与 `released.json`；
  **不采六帧、不下判据②–⑥结论**。
- 退出码：`0` = 已进入 5 px 容差；`3` = 测量有效但**尚未对齐**（不是脚本错误，
  继续 §3.1 调光路）；`2` = 等不到台面锁；`1` = 其它错误。
- **放行门（缺一不放行）**：
  1. `alignment_measurement.json` 的 `within_tolerance=true` 且
     `mean_offset_from_reference.dist_px ≤ 5.0`；
  2. `any_frame_saturated=false`（任一平场帧饱和 ⇒ 本次质心不可信，
     按光路纪律加衰减/降光强后重测；不得改 0.087 ms 曝光、12-bit 或相位）；
  3. `released.json` 的 `slm_closed` / `camera_closed` / `lock_released` /
     `closed_normally` 全 true、`error=null`。
  任一条不满足 ⇒ 回到 §3.1 继续调光路，**不把未达标的探针目录当成功证据**。
- 离线验收（2026-11-27，未开设备）：`18 passed`（定向 pytest，EXITCODE=0）、
  `py_compile` EXITCODE=0、`--self-test` 输出
  `{"self_test": "passed", ...}` EXITCODE=0、`ruff check --extend-select RUF100`
  `All checks passed!` EXITCODE=0、plan 模式运行前后目录均不存在（无副作用）。
- 每段探针结束按 §4 回填设备占用/释放报告；开下一段前重新核算 SLM 200 s 冷却。

### 3.3 六帧重采（探针放行后，全新目录）

```bash
python scripts/slm/capture_s1_mapping_disambiguation_20261124.py \
  --execute \
  --out data/debug/slm_transfer/s1_verify_20261127_after_alignment \
  --wait-until-idle 1800
```

- `--out` 必须是**全新不存在**目录（`exist_ok=False`）；作废的
  `s1_verify_20261126_retry/` **保留不删、不覆盖**（原始数据纪律）。
- 采集期间不判读、不并行占用、不采 dark、不操作激光。
- 采完先确认 `summary.status=ok`、6 帧齐全、0 饱和、`released.json` 正常关闭，
  再读 `flat_centroid_offset_px`：若仍 >5 px ⇒ 判据①再次证伪，整组作废、不得进入判据②–⑥。

### 3.4 判读（仅在判据①通过后）

- 判据②–⑥按 `PREREG_CRITERIA_VERBATIM` 逐条判读，阈值一个字不改：
  x 光栅 ±1 级列向 56–59 px / 行向残差 ≤3 px；scale 对 ±5% 能量门；
  ladder 95 px 整数倍 ±2 px；blaze2d 单斑；任一帧饱和整体 abort。
- 结论只用三档：**已证实有效 / 已证实无效 / 证据不足**。预测指标改善不算实测算法收益；
  局部标定不成立不能写成物理机制成立。

## 4. 设备占用/释放报告（每次探针/采集后逐项回填）

| 项 | 值 |
|---|---|
| 占用内容 | SLM #1 + 大恒 cam 0；`--wait-until-idle 1800`；未采 dark、未操作激光、未并行占用 |
| 占用起（UTC） | 从 `plan.json` / `summary.json` 回填 |
| 占用止（UTC） | 从 `summary.json` 回填 |
| 释放（UTC） | 从 `released.json` 回填 |
| 释放状态 | `slm_closed` / `camera_closed` / `lock_released` / `closed_normally` 全 true、`error=null` ⇒ 正常关闭 |

### 6.3 光路恢复探测（2026-11-27 16:12 本地，`optical_path_probe/a1_recovery_20261127/20261127_081158Z`）

- 执行 `python scripts/slm/probe_optical_path_validity.py --session a1_recovery_20261127 --wait-until-idle 1800`，
  verdict `state=ok`、退出码 `0`。曝光阶梯：0.05 ms 峰值 1571、0.2 ms 峰值 3869（未触 90% 线）、
  1 ms 起饱和；离焦 +1 rad 峰值变化 30.2%、宽度变化 25.9%（>3×合并标准误）⇒ **光路有效、SLM 在调制**。
- 同一次探测的平场 0 级质心 `(1104.42, 1228.01)`，对 A1 参考 `(1105.249, 1227.556)`
  距离 **0.94 px**。
- 所以 12:57 的"无 0 级信号"是**当时确有光路中断**（峰值 40 ≈ 噪声底），不是相位/脚本问题；
  16:11 复测时光已恢复。两次探测之间我未做任何光路操作。

### 6.4 对齐探针重跑（2026-11-27 16:17 本地，`a1_align_probe_02`，全新目录）

- 冷却：上次 SLM 关闭 `16:13:47` 本地，等待满 200 s 后启动，实测间隔 223 s。
- 退出码 3、`status=ok`、三帧 `max=2400/2404/2315`（未饱和、远高于噪声底 20.9）、
  preflight peak 2443。**0 级光斑清晰可见，峰值量级恢复**（11-26 为 3512）。
- 脚本报 `mean_centroid_px=(1003.10, 1275.50)`、`dist_px=112.836`、`within_tolerance=false`。

### 6.5 根因（已证实无效）：`_centroid_px` 被背景噪声主导，不是光斑质心

- `_centroid_px`（`capture_s1_mapping_disambiguation_20261124.py:358`）用**全帧中位数**做基线，
  所有正残差像素全部计入权重、无任何掩模。全帧 5,038,848 像素里光斑只占约 1400 像素，
  背景噪声像素数约 5.03 M，每像素正残差期望 ≈0.4 counts ⇒ 噪声权重与光斑权重同量级甚至更大。
- 定量分解（把权重按 `frame >= median+6σ` 拆成 spot / noise 两部分，再按权重合成）：

  | 帧 | 光斑权重 @ 位置 | 噪声权重 @ 位置 | 噪声占比 | 合成预测 | 脚本实测 | 残差 |
  |---|---|---|---|---|---|---|
  | 11-26 S1 flat | 1,238,585 @ (1106.20, 1228.39) | 1,937,400 @ (963.07, 1301.59) | 61.0% | (1018.886, 1273.041) | (1018.886, 1273.041) | **0.0000 px** |
  | 11-27 f0 | 794,774 @ (1104.86, 1228.32) | 2,025,460 @ (966.13, 1293.80) | 71.8% | (1005.224, 1275.346) | (1005.224, 1275.346) | **0.0000 px** |
  | 11-27 f1 | 765,647 @ (1104.90, 1228.15) | 1,995,152 @ (961.80, 1294.44) | 72.3% | (1001.482, 1276.058) | (1001.482, 1276.058) | **0.0000 px** |
  | 11-27 f2 | 763,427 @ (1105.06, 1227.86) | 2,005,532 @ (963.61, 1293.07) | 72.4% | (1002.608, 1275.093) | (1002.608, 1275.093) | **0.0000 px** |

  ⇒ 脚本输出的"偏移 97.6 / 112.8 px"**算术上就是光斑与背景噪声的加权混合**，残差 0.0000 px；
  它既非光斑位移，也非束流差异。噪声质心落在 (963, 1294) 附近（≈ 帧几何中心 971.5/1295.5），
  所以输出恒被拉向帧中心约 100 px。
- **同一批帧换任何合理掩模估计，0 级都在 A1 参考上（已证实）**：

  | 帧 | argmax | 3σ | 4σ | 6σ | 10σ | 20σ | ROI40 | ROI80 |
  |---|---|---|---|---|---|---|---|---|
  | 11-26 S1 flat | (1107, 1229) | 0.90 | 1.10 | 1.26 | 1.31 | 1.38 | 1.36 | 1.19 |
  | 11-27 f0 | (1105, 1229) | 1.62 | 1.14 | 0.86 | 0.81 | 0.78 | 0.77 | 0.46 |
  | 11-27 f1 | (1105, 1228) | 1.65 | 1.01 | 0.69 | 0.71 | 0.61 | 0.63 | 0.47 |
  | 11-27 f2 | (1106, 1228) | 1.16 | 0.73 | 0.36 | 0.41 | 0.36 | 0.30 | 0.41 |

  表内为到 A1 参考的欧氏距离（px），全部 ≤5 px。与 11-23 报告的口径一致
  （`agent_notes_20261123.md`：0 级位置 (1106, 1229)）。
- 结论：判据①**按现行实现不可满足**——要让噪声占比降到 5% 以下需光斑权重 ≈38 M counts，
  约为现值的 48 倍，而 0.087 ms 下再提高只能饱和。故"调光路"无法让它通过；
  11-26 的"不同束流"归因是**该测量函数的假阴性**，此前记录作废。
- 影响范围：`_centroid_px` 只用于判据①（`capture_s1_mapping_disambiguation_20261124.py:525`），
  判据②–⑥ 用的是落盘原始帧，不受影响。对齐探针 `probe_a1_zero_order_alignment_20261127.py`
  直接复用了它，因此同样继承该缺陷。

## 5. 边界

- 不改 S1 采集脚本、不改冻结判据/测试、不复用作废帧、不事后调阈值、不修改原始数据。
- 不 push；若要提交，继续用临时索引只暂存本车道 blob（探针脚本、离线测试、本文）。
- 探针/采集脚本与离线测试在写作时未实验执行；`--execute` 只在用户授权后运行。

## 6. 执行记录

### 6.1 对齐探针首跑（2026-11-27 12:57 本地，`a1_align_probe_01`）

- 授权后执行 `--execute --wait-until-idle 1800`，退出码 3（测量有效但超差）。
- 前置：台面锁互斥字节 0、无 SLM/CCD 进程、上次 S1 释放 `2026-11-26T18:58` 本地且
  `closed_normally=true`，冷却 ≫200 s。
- **结果：CCD 无 0 级信号 ⇒ 证据不足，不重采六帧。** preflight peak=40 counts、
  三帧 max=38/43/38，无任何像素 >100 counts（满量程 4095 的 ~1%），argmax 落在
  角落 (1,460) 杂散。均值质心 (956.75, 1311.26) / dist=170.47 px 是背景矩伪迹，
  **不是真实光斑位置**，既不能判"进容差"也不能判"不同束流"。
- 11-26 S1 同图样、同 0.087 ms 曝光峰值 3512 counts；同 flat 灰度哈希
  (`cb228af4…`)。差异指向光路无光/被遮/激光未开或曝光未生效，**属光路/激光层问题，
  不是相位公式、不是脚本、不是对齐位置偏 170 px**。
- 探针缺少"无信号 fail-closed"门：只有饱和上限（0.9×4095），没有信号下限，
  所以无光时仍返回"测量有效但超差(rc=3)"而非明确的 `no_signal`。**这是探针自身的
  缺口，本次不改脚本**（改脚本须走离线测试+静态门禁流程）；操作员须以 raw 峰值
  自行判信号有无。
- 处置：先按 `ao-optical-path-calibration` 六步探测恢复光路，确认有 0 级光斑、
  peak 回到合理量程（参考 11-26 的 3512，且不饱和）后，再用**全新目录**重跑探针。
  不据此采六帧、不改阈值、不改 S1 脚本。

### 6.2 设备占用/释放报告（探针首跑，逐项回填）

| 项 | 值 |
|---|---|
| 占用内容 | SLM #1 + 大恒 cam 0；`--wait-until-idle 1800` 排队拿锁；未采 dark、未操作激光、未并行占用 |
| 占用起（UTC） | `2026-11-27T04:56:52.711696Z` |
| 占用止（UTC） | `2026-11-27T04:57:14.628091Z` |
| 释放（UTC） | `2026-11-27T04:57:14.628118Z` |
| 释放状态 | `slm_closed` / `camera_closed` / `lock_released` / `closed_normally` 全 true、`error=null` ⇒ **正常关闭** |

### 6.7 设备占用/释放报告（光路探测 + 探针重跑，2026-11-27 下午）

| 项 | 光路探测 | 对齐探针重跑 |
|---|---|---|
| 占用内容 | SLM #1 + 大恒 cam 0（曝光阶梯 8 档 + 离焦 5 图样）；未采 dark、未操作激光、未并行占用 | SLM #1 + 大恒 cam 0（1 图样 x 3 帧）；未采 dark、未操作激光、未并行占用 |
| 占用起（UTC） | `2026-11-27T08:11:58Z` | `2026-11-27T08:17:31.922Z` |
| 占用止（UTC） | `2026-11-27T08:13:47Z` | `2026-11-27T08:17:44.953Z` |
| 释放（UTC） | `2026-11-27T08:13:47Z` | `2026-11-27T08:17:44.953Z` |
| 释放状态 | record `occupancy.closed_normally=true` | `slm_closed` / `camera_closed` / `lock_released` / `closed_normally` 全 true、`error=null` |
| 冷却 | — | 距上次 SLM 关闭 223 s（>=200 s）后才 open |

### 6.8 判据① 测量口径 v2（2026-11-27 17:00 本地，用户裁定选项 A）

- 用户裁定：**修测量函数 + 离线判六帧**。冻结判据① 的参考点 `(1105.249, 1227.556)`
  与 5 px 容差**一个字不改**；只把取点函数从"全帧无掩膜基线质心"改为"信号掩膜质心"，
  并登记为判据① v2 口径（ summary 新增 `flat_zero_order_criterion.estimator_revision="v2"`、
  `measured` / `dist_px` / `within_tolerance` / `measurement` 字段）。
- 新量测（`capture_s1_mapping_disambiguation_20261124.py:flat_signal_measurement`）：
  `sigma=max(1.4826×MAD, 1 count 量化地板)`，掩膜 `frame >= median + 6σ`，
  权重 `max(frame-median,0)`，质心只在掩膜像素上算；无信号门
  `peak_snr≥50 且 signal_pixels≥4 且 signal_weight≥2000 counts`。
  无信号时 `centroid_px` 退到几何中心并置 `centroid_is_fallback=true` /
  `has_signal=false`，**不把噪声簇冒充测量值**；冻结字段
  `same_session_flat_centroid_px` 仍恒有值以维持既有 summary 契约。
- 11-26/11-27 冻结帧 v2 复算（`6σ` 掩膜）：11-26 flat `(1106.20, 1228.39)` dist **1.26 px**、
  f0 `(1104.86, 1228.32)` 0.86 px、f1 `(1104.90, 1228.15)` 0.69 px、
  f2 `(1105.06, 1227.86)` 0.36 px；全部 `has_signal=true`；12:57 无光帧
  `peak_snr=16.2`、`has_signal=false`，正确 fail-closed。
- 离线验证（无设备、无台面锁）：
  - `.venv\Scripts\python.exe -m pytest tests/ao_shaping/scripts/test_s1_zero_order_measurement_v2_20261127.py tests/ao_shaping/scripts/test_capture_s1_mapping_disambiguation_20261124.py tests/ao_shaping/scripts/test_probe_a1_zero_order_alignment_20261127.py -q --import-mode=importlib` →
    **36 passed, EXITCODE=0**（新增 7 项 v2 测试；18 项对齐探针测试与 11 项冻结采集测试均未改、全过）。
  - `.venv\Scripts\python.exe scripts/slm/capture_s1_mapping_disambiguation_20261124.py --self-test` →
    `{"self_test":"passed","n_frames":6}` EXITCODE=0；对齐探针 `--self-test` 同样 EXITCODE=0。
  - `.venv\Scripts\ruff.exe check <上述 4 个 py 文件> --extend-select RUF100` → All checks passed, EXITCODE=0；
    术语门禁 `python scripts/_common/term_unify_20261016.py --root <上述 py>` → 0 文件/0 行 EXITCODE=0；
    新增/修改文件 BOM=False、CRLF=0。
- 对齐探针仍直接复用 `_centroid_px`，因此同一 v2 口径自动生效；本轮**没有再占用设备**，
  台面锁互斥字节复核为 0。

### 6.9 既有 11-26 六帧离线判读②–⑥（2026-11-27 17:29 本地）

- 产物：`scripts/slm/judge_s1_criteria_offline_20261127.py`、
  `report/shape_trials/s1_offline_judge_20261127/judgment.json` / `judgment.md`；
  已按 `report/AGENTS.md` 登记 `scripts/_common/provenance.py::REPORTS` 并重生成
  `report/README.md` 索引。全程只读六帧，不操作 SLM / CCD / 激光、不申请台面锁。
- 位移原点：flat 帧 v2 掩膜质心 `(1106.2005, 1228.3862)`、`has_signal=True`；
  判据① 未重判。斑点检测复用冻结 `find_spots()`，平顶重复峰按 peak 降序 +
  Chebyshev ≤3 px 贪心合并，合并规则写入证据 JSON。
- 逐条裁定（阈值/文本一个字未改）：

| 判据 | 裁定 | 量化证据 |
|---|---|---|
| ② | 已证实无效 | grating_x ±1 列位移 `+95.0929 / -93.2201 px`，冻结窗 `56–59 px`；+1 行残差 `3.7212 px > 3 px`。冻结窗与 x 光栅实测冲突（y 控制帧约 59 px），不得自行改阈值。 |
| ③ | 已证实有效（判 1024） | phase_sha256 相同、gray_sha256 不同；R=5/8/12/16 px × 两种中心口径共 8 组窗口，ratio(1024/993)=`1.083–1.108`，全部 `>1.05`，无符号冲突。 |
| ④ | 已证实无效 | 冻结阶次 `LADDER_ORDERS=(1,3)` ⇒ 预期 `95 / 285 px`；主斑实测 `88.3199 / 231.8747 px`，残差 `6.6801 / 53.1253 px > ±2 px`；`106.7774 px/peak 56` 为未归属弱斑。 |
| ⑤ | 证据不足 | 单斑、符号象限匹配；预测 `(-59.8121, +39.963)`，实测 `(-56.7783, +43.5669)`，残差 `4.7108 px`；冻结文本无数值容差，不自设阈值判过/判死。 |
| ⑥ | 已证实有效 | 六帧 max `3512/1702/1857/1785/1776/2334 counts`、preflight `1698`，均 `<0.9×4095=3685.5`，`abort=false`。 |

- 总体裁定：**已证实无效**（②、④ 越窗/越容差）。这是预注册判据口径下的结论，
  不是"算法预测指标改善"，也不因该结论声称任何整形收益。②/④ 的失败量级远超噪声，
  现有帧足以判死，**不据此重新实验重采**。
- 门禁：`judge --self-test` EXITCODE=0；真实判读 EXITCODE=0；`ruff check --extend-select RUF100 <新脚本>` EXITCODE=0；
  冻结测试 11 passed EXITCODE=0；`py_compile` / 术语门禁 / BOM / LF 均过。报告溯源登记后
  `sync_report_provenance.py` 仅新增本报告头与索引行（其余未注册报告属既有状态）。

---

Done — 判据① 已按用户裁定的 v2 掩膜质心口径修正并备案；既有 11-26 六帧②–⑥ 已纯离线判读完毕，未重新实验。
Verified — ① v2 dist 1.26 px（<=5）；②/④ 已证实无效，③/⑥ 已证实有效，⑤ 证据不足；36 项相关 pytest + 11 项复跑均过，ruff/术语/溯源/BOM/LF 全过；本轮 SLM / CCD / 激光未占用，台面锁为 0。
Remaining — 判据②冻结窗 56–59 px 与 x 光栅实测约 93–95 px 冲突；判据⑤缺冻结容差；探针无信号 fail-closed 门未授权。
Needs — 用户对判据②窗口冲突和判据⑤容差的裁定；未获授权不 push，不自行改冻结文本或阈值。
