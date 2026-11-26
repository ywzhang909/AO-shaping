# S1e 修复后重采·离线准备报告（2026-11-26）

<!-- provenance:start -->
> **预注册**: [`docs/dev/s1e_postfix_prereg_20261126.md`](../../../docs/dev/s1e_postfix_prereg_20261126.md)
> **修复脚本**: [`src/ao_shaping/drivers/slm/santec/driver.py`](../../../src/ao_shaping/drivers/slm/santec/driver.py)（`_write_to_memory` §12.16.5 守卫）
> **采集脚本**: [`scripts/slm/s1e_same_bytes_probe_capture_postfix_20261126.py`](../../../scripts/slm/s1e_same_bytes_probe_capture_postfix_20261126.py)
> **判读脚本（冻结，复用）**: [`scripts/slm/s1e_same_bytes_probe_judge_20261126.py`](../../../scripts/slm/s1e_same_bytes_probe_judge_20261126.py)
> **绑定 commit**: 代码 / 测试 / 预注册见 `s1e_postfix_code` 提交；本报告在其后提交，以其 commit id 为准（见 §6）。
> **运行环境**: 全程**离线**。本轮未占用 SLM/CCD/激光、未取设备锁、未建 run 目录；实验须用户另行授权。
<!-- provenance:end -->

## 摘要

本轮完成 S1 归因线根因 `non_contiguous_memmove_transpose` 的**离线等价修复与全部门禁**：在 `driver.py::_write_to_memory` 的 `ctypes.memmove` 前加 `np.ascontiguousarray` 连续性守卫，使 F-contiguous 的 H 光栅按逻辑 C 序送屏（物理字节 `b98be9a6…` → `7276d9da…`），C-contiguous 的 B/A/flat 零改动。新增两条回归门禁（driver 守卫 + 修复后采集脚本）与一份修复后采集脚本（冻结口径逐字不变，仅把 `physical_sha256` 改记实际送屏字节、另记 `pre_guard_buffer_sha256` 供审计）。定向 pytest **42 项全过**，ruff / BOM / 计划模式 / 冻结 blob 核验均通过。冻结脚本、冻结测试、冻结判据、冻结测试集与变体 A/P 原始记录**未改**。预注册与实验清单已成文；**本轮不实验**，S1 维持 `BLOCKED`。

---

## 1. 背景与判据链

| 阶段 | 结论 |
|---|---|
| S1b | `insufficient_consistency`（内容轴 2×2 一致性不足）|
| S1c | `session_nondeterminism`（会话间不确定）|
| S1d | `delivery_nondeterministic_same_bytes`（同字节流仍不同远场，J1 不过）|
| S1e A（run_01）| D1 `D-fail`、E2；单次 open/close |
| S1e P（run_02）| 物理断电重启后重采；D1 `D-fail`（H 两两距离 1.585 px > 1.5）、E4 不触发、维持 E2；`no_power_cycle_effect` |

§12.16.4 结论：所有分支 S1 `BLOCKED`，不做 D bracket，G1/G3/G4 暂缓；只有修根因后重采取得 D-pass 才可能解锁。

## 2. 根因确认

- H 灰度为 F-contiguous `(2,2400)`、strides `(2,2400)`；裸 `memmove` 按物理列主序拷贝，面板按行主序 `(1200,1920)` 解释 → 周期 128→≈80 px、`128/80=1.6=1920/1200`、条纹轴列变行，与 S1d 指纹一致。
- 代码路径：`display_data → _write_phase(L1468) → _write_to_memory(L2660) → memmove`（`video_mode=0`）；另有 `_display_video(L1524)` 的 memmove 不在本采集路径。
- 排除 SDK/固件/OS 旋转（探测无旋转）。

## 3. 修复与唯一变量

```python
# driver.py::_write_to_memory, memmove 之前
phase = np.ascontiguousarray(phase)
```

送屏字节变化（门禁断言）：

| array | 逻辑 gray SHA | 修复前物理 | 修复后实际送屏 |
|---|---|---|---|
| H | `7276d9da…` | `b98be9a6…` | `7276d9da…` |
| B | `c8d937ce…` | `c8d937ce…` | `c8d937ce…` |
| flat | `cb228af4…` | `cb228af4…` | `cb228af4…` |

修复后采集脚本的 `physical_sha256` = 实际送屏字节（judge 分组字段），`pre_guard_buffer_sha256` = 修复前裸 buffer（审计）。冻结判据 `FROZEN_CRITERIA`、图样顺序、时序参数逐字不变。

## 4. 门禁结果（离线）

| 门禁 | 结果 |
|---|---|
| 定向 pytest（守卫 + 修复后采集 + 冻结采集 + driver）| **42 passed / 24.90 s** |
| ruff check（4 本线文件）| `All checks passed!`（EXIT=0）|
| BOM | 4 文件全 `no-BOM` |
| 采集计划模式 | 12 pattern / 24 frame；H phys `7276d9da…`、pre-guard `b98be9a6…`、B `c8d937ce…`；阈值与冻结一致 |
| 冻结 blob | capture `640d85c6…`、test `80de251d…`，`git diff HEAD` 为空 |
| 术语 `--check` | 本线文件不在漂移清单；仓库 60 文件/1024 行为他车道基线 |

## 5. 指标参数与实验清单

冻结采集参数：SLM #1 Memory 模式、大恒 cam 0 Mono12/full 4095、1064 nm、0.087 ms、settle 0.8 s、`G_REF=1023`、饱和门 `0.9×4095`、`focus_d_m1=0.0`、12×2=24 帧、`--wait-until-idle 1800`。

实验（用户授权后执行，本轮不做）：

```bash
python scripts/slm/s1e_same_bytes_probe_capture_postfix_20261126.py --execute \
  --out data/experiments/attribution_20261123/s1/s1e_same_bytes_probe_20261124/run_03_postfix \
  --wait-until-idle 1800
```

前置：SLM #1 枚举正常、距上次释放（`2026-11-26T06:13:37Z`）充分冷却、CCD 空闲、无并行占用。采集期间不判读、不采 dark、不操作激光。释放须 `slm_closed`/`camera_closed`/`lock_released`/`closed_normally` 全 `true`、`error=null`。判读复用冻结 judge，正式判读报告在释放确认后产出。完整清单见预注册 §5。

## 6. 任务项

- [x] 根因定位与唯一变量确定
- [x] 连续性守卫修复 + 离线回归（42 passed）
- [x] 修复后采集脚本 + 计划模式自检 + 冻结 blob 核验
- [x] 预注册 + 实验清单（离线）
- [x] 本报告（离线准备）
- [ ] 用户授权后按 §5 实验（**本轮不做**）
- [ ] 释放确认后离线判读 + 正式判读报告（**实验后做**）

## 7. 结论与边界

本轮只完成离线准备工作，**未取得新实验数据**。S1 维持 `BLOCKED`；冻结口径、冻结测试集、A/P 原始记录不动。实验与判读是后续独立步骤，须用户授权并据释放状态与 D1–D6 结果另行结论。
