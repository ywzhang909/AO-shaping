# S1e 修复后重采预注册（post-fix round，2026-11-26）

<!-- provenance:start -->
> **设计源**: [`hw_window_20261123_attribution_prereg.md`](hw_window_20261123_attribution_prereg.md) §12.16（冻结口径，blob 不变）
> **现行计划**: [`next_experiment_plan_20261124.md`](next_experiment_plan_20261124.md) §1 / §2.P1
> **修复门禁测试**: [`test_santec_write_to_memory_contiguity.py`](../../tests/ao_shaping/drivers/slm/test_santec_write_to_memory_contiguity.py)
> **修复后采集脚本**: [`s1e_same_bytes_probe_capture_postfix_20261126.py`](../../scripts/slm/s1e_same_bytes_probe_capture_postfix_20261126.py)
> **修复后采集测试**: [`test_s1e_same_bytes_probe_capture_postfix_20261126.py`](../../tests/ao_shaping/scripts/test_s1e_same_bytes_probe_capture_postfix_20261126.py)
> **判读脚本（冻结，复用）**: [`s1e_same_bytes_probe_judge_20261126.py`](../../scripts/slm/s1e_same_bytes_probe_judge_20261126.py)
> **运行环境**: 本文与所有改动、门禁、计划模式下机清单均为**离线**产出，本轮**未占用 SLM/CCD/激光**；实验须经用户另行授权后按 §5 清单执行。
> **绑定 commit**: 预注册/修复/测试/脚本在本轮提交后以其 commit id 为准（见 §7）。
<!-- provenance:end -->

## 摘要

- **目标**：在修复 `non_contiguous_memmove_transpose` 根因后，按冻结 §12.16 口径重采一次 S1e「同一字节流 ⇒ 同一远场」探针，检验 D1（同 array 多上传主瓣两两距离 ≤1.5 px）是否由 `D-fail` 翻转为通过——这是 S1 从 `BLOCKED` 解锁的**唯一**技术路径。
- **唯一变量**：`driver.py::_write_to_memory` 在 `ctypes.memmove` 前对非 C-contiguous 相位加 `np.ascontiguousarray` 守卫；送屏字节流对 H 从 F 序裸 buffer `b98be9a6…` 变为逻辑 C 序 `7276d9da…`。采集脚本、判读脚本、冻结判据、时序参数、图样顺序与 A/P 原始记录均不变。
- **本轮产出**（离线）：根因修复 + 连续性回归门禁（42 项定向 pytest 全过）、修复后采集脚本与计划模式自检、本文预注册与实验清单。**不操作设备**。
- **判读口径**：仍走冻结 judge 的 D1–D6 与 E1–E4；D-pass 才可能解锁 S1，任何结果下**本轮不升格 S1、不做 D bracket、G1/G3/G4 继续暂缓、冻结测试集不动**。

---

## 1. 根因与修复（唯一变量）

### 1.1 根因链（§12.16.7 已确认）

| 环节 | 事实 |
|---|---|
| 现象 | S1d：`delivery_nondeterministic_same_bytes`（J1 不过）；S1e A/P 同字节但仍 D-fail、E2；H 远场周期约 80 px（应为 128），条纹轴列变行 |
| 算术指纹 | `128/80 = 1.6 = 1920/1200`（面板形状与外部解释形状互换）|
| 内存布局 | H 灰度 `grating_phase(axis=1)` 物化为 F-contiguous `(2,2400)`，strides `(2,2400)`；B/A/flat 为 C-contiguous |
| 代码路径 | `display_data` → `_write_phase`(L1468) → `_write_to_memory`(L2660) → `ctypes.memmove(dat, phase.ctypes.data, phase.nbytes)`；`video_mode=0`（Memory）|
| 根因 | 裸 `memmove` 按物理内存序拷贝，列主序 H 字节被面板按行主序解释 → 图案转置。非 SDK/固件/OS 旋转；`EnumDisplaySettings` 探测无旋转 |

### 1.2 修复（已落盘、离线验证）

`src/ao_shaping/drivers/slm/santec/driver.py::_write_to_memory` 在 `memmove` 前插入：

```python
phase = np.ascontiguousarray(phase)
```

- 仅对非 C-contiguous 输入改变送屏字节；C-contiguous（B/A/flat）**零改动**。
- 缓存 `_memory_phase_cache[memory_number] = phase.copy()` 随之写入 C 序副本，与实际上屏一致。
- **未触碰** `driver.py` 中他车道 `SlotRotator.wait_time_s` 透传 hunk（仍留在工作树，属持锁车道）。

### 1.3 摘要变化（门禁断言）

| array | 逻辑 gray SHA（不变）| 修复前物理 buffer | 修复后实际送屏（`physical_sha256`）|
|---|---|---|---|
| H | `7276d9da…` | `b98be9a6…`（F 序裸 buffer）| `7276d9da…`（== 逻辑 C 序）|
| B | `c8d937ce…` | `c8d937ce…` | `c8d937ce…`（不变）|
| flat | `cb228af4…` | `cb228af4…` | `cb228af4…`（不变）|

修复后采集脚本把修复前的 F 序裸 buffer 另行记为 `pre_guard_buffer_sha256`，仅用于审计，`physical_sha256` 一律记实际送屏字节，供冻结 judge 的同字节分组使用。

---

## 2. 冻结口径（逐字继承，不重定义）

- **数组/顺序/前驱**：§12.16.2 冻结的 12 图样顺序不变；H 在索引 1,2,4,6,8,10；H/B 相位与 gray 摘要不变（`d8f16e03…/7276d9da…`、`382100b0…/c8d937ce…`）。
- **判据阈值**（`FROZEN_CRITERIA`，冻结脚本与 judge 共用）：`intra_upload_max_dist_px=1.5`；D1 `{min_effective_uploads:5, max_pairwise_dist_px:1.5}`；D2 `{axis_ratio_min:1.5, col_dist_px:[90,102], row_dist_px:[52,64]}`；D4 `{weak_peak_fraction:0.10}`；D6 `{zero_exception_groups_required:True}`。
- **冻结 blob（相对 HEAD 不得变化）**：采集脚本 `640d85c6d786dc5518562c1909369cd4c42ad47b`、冻结测试 `80de251d1ce16eab12b3debfb811bdbee1dc6715`；冻结 judge `s1e_same_bytes_probe_judge_20261126.py`。本轮全部未改（已用 `git diff HEAD` 与 `git hash-object` 核验）。
- **不动**：S1b/S1c/S1d 与变体 A/P 原始记录、冻结测试集、`hw_window_20261123_attribution_prereg.md`（他车道改动，未触碰）、`scripts/_common/provenance.py`（他车道混杂改动，未触碰）。
- **改口径/改驱动属新一轮**：本轮即该新一轮；重采数据写入全新 run 目录，不覆盖 `run_01`/`run_02`。

---

## 3. 门禁证据（离线，已执行）

| 门禁 | 命令 | 结果 |
|---|---|---|
| 定向 pytest | `.venv\Scripts\python.exe -m pytest --import-mode=importlib -q tests/.../test_santec_write_to_memory_contiguity.py tests/.../test_s1e_same_bytes_probe_capture_postfix_20261126.py tests/.../test_s1e_same_bytes_probe_capture_20261124.py tests/.../test_slm_shift.py` | **42 passed** |
| ruff | `.venv\Scripts\python.exe -m ruff check <4 个本线文件>` | `All checks passed!`（EXIT=0）|
| BOM | 四文件读首 3 字节 | 全 `no-BOM` |
| 采集计划模式 | `python scripts/slm/s1e_same_bytes_probe_capture_postfix_20261126.py`（不带 `--execute`）| 12 pattern / 24 frame；H 物理 `7276d9da…`、pre-guard `b98be9a6…`、B `c8d937ce…`；判据与冻结一致；只读、不取锁、不建目录 |
| 冻结 blob | `git hash-object` + `git diff HEAD` | 采集/测试哈希与 HEAD 一致，diff 为空 |
| 术语 | `term_unify_20261016.py --check` | 本线 4 文件均不在漂移清单；仓库 60 文件/1024 行漂移为他车道基线（非本轮引入）|

> 说明：Windows 系统 Python 的 numpy 与本工程 ABI 不合（cp314 vs cp315），一律用 `.venv\Scripts\python.exe`。PowerShell 将 loguru stderr DEBUG 包装成 `NativeCommandError` 不是真实错误，以 `EXITCODE=0` + `summary.status=ok` + 四释放标志为准。

---

## 4. 指标参数（采集，冻结值）

| 参数 | 值 |
|---|---|
| SLM | #1，Memory 模式 `video_mode=0`，驱动板枚举须正常 |
| CCD | 大恒 cam 0，全帧 Mono12，full scale 4095 |
| 波长 / 曝光 / settle | 1064 nm / 0.087 ms / 0.8 s |
| 编码 | `G_REF=1023`（`ENCODING_MAX_GRAY`），饱和门 `0.9×4095` |
| 焦距补偿 | `focus_d_m1=0.0`（不施加）|
| 图样 | 12 上传 × 2 帧 = 24 帧；暗场不采；激光不操作 |
| 设备锁 | `--wait-until-idle 1800`；单会话一次 open/close；不并行占用 |

---

## 5. 实验清单（用户授权后执行；本轮不执行）

### 5.1 前置确认（缺一不采集）

1. SLM #1 枚举正常（先做桌面页加载探测：只枚举、不开相机/设备）；驱动板 ID 可读。
2. 距上次释放（`2026-11-26T06:13:37Z`）已充分冷却（≫200 s）。
3. 大恒 cam 0 空闲；无其他会话占用 SLM/CCD/激光。

### 5.2 采集命令（全新 run 目录，逐字执行）

```bash
python scripts/slm/s1e_same_bytes_probe_capture_postfix_20261126.py \
  --execute \
  --out data/experiments/attribution_20261123/s1/s1e_same_bytes_probe_20261124/run_03_postfix \
  --wait-until-idle 1800
```

- `--out` 必须是**全新且不存在的目录**；脚本以 `exist_ok=False` 创建，存在即失败，不会覆盖 A/P 记录。
- 采集期间**不判读、不并行占用、不采 dark、不操作激光**。

### 5.3 设备占用/释放报告（采集结束后逐项回填）

| 项 | 值 |
|---|---|
| 占用内容 | SLM #1 + 大恒 cam 0；`--wait-until-idle 1800`；未采 dark、未操作激光、未并行占用 |
| 占用起（UTC） | `started_at_utc=<从 summary.json 回填>` |
| 占用止（UTC） | `finished_at_utc=<从 summary.json 回填>` |
| 释放（UTC） | `released_at_utc=<从 released.json 回填>` |
| 释放状态 | `slm_closed` / `camera_closed` / `lock_released` / `closed_normally` 须**全 true**、`error=null` ⇒ 正常关闭 |
| 采集状态 | `EXITCODE=0`、`summary.status=ok`、24 帧全采齐、0 饱和 |

### 5.4 判读（释放确认后，离线）

```bash
python scripts/slm/s1e_same_bytes_probe_judge_20261126.py \
  --capture-dir data/experiments/attribution_20261123/s1/s1e_same_bytes_probe_20261124/run_03_postfix \
  --out-dir report/shape_trials/s1e_postfix_20261126 \
  --stem s1e_postfix_20261126
```

- 复用冻结 judge，D1–D6 与 E1–E4 口径不变；`physical_sha256` 现记实际送屏字节，judge 按 `sent_array` 分组无需改脚本。
- 正式报告写入 `report/shape_trials/s1e_postfix_20261126/`，简体中文、含摘要/图表/任务项/指标参数与 provenance 头、附 commit id。

---

## 6. 判读与解锁条件

| 结果 | 处理 |
|---|---|
| D-pass（D1 等全过，主瓣两两距离 ≤1.5 px）| S1 具备**解锁候选**条件；按 §12.16.4 走正式解锁流程（另行预注册），本预注册**不直接升格** S1、不做 D bracket |
| D-fail / E2 任何分支 | S1 继续 `BLOCKED`；记录新归因，轴/K 冻结与 G1/G3/G4 继续暂缓 |
| 任何结果 | 冻结测试集不动；A/P 原始记录不改写；本轮不跳过判据、不重定义阈值 |

---

## 7. 边界与提交

- 本线改动文件（逐路径 `git add`，仅本线）：
  - `src/ao_shaping/drivers/slm/santec/driver.py`（**仅 §12.16.5 守卫 hunk** 以 cacheinfo 法暂存；`SlotRotator` hunk 留持锁车道）
  - `scripts/slm/s1e_same_bytes_probe_capture_postfix_20261126.py`（新增）
  - `tests/ao_shaping/drivers/slm/test_santec_write_to_memory_contiguity.py`（新增）
  - `tests/ao_shaping/scripts/test_s1e_same_bytes_probe_capture_postfix_20261126.py`（新增）
  - `docs/dev/s1e_postfix_prereg_20261126.md`（新增，本文）
  - `docs/dev/next_experiment_plan_20261124.md`（§2.P1 加一行状态指针；该文件 HEAD 干净、无他车道改动）
- **不碰**：`scripts/_common/provenance.py`、`hw_window_20261123_attribution_prereg.md`、`driver.py` 的 `SlotRotator` hunk、冻结脚本/冻结测试/冻结 judge。
- 提交后 commit id 写入 §7 与实验报告 provenance 头。

### 任务项

- [x] 根因修复 + 连续性守卫（离线）
- [x] 离线回归门禁（42 passed / ruff / BOM / 计划模式 / 冻结 blob 核验）
- [x] 修复后采集脚本 + 测试 + 自检
- [x] 预注册 + 实验清单（本文，离线）
- [ ] 用户授权后按 §5 实验（**本轮不做**）
- [ ] 释放确认后离线判读 + 正式报告（**实验后做**）
