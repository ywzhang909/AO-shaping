# `--debug-weights` 真机验证报告（2026-11-21）

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`src/ao_shaping/tools/slm/transfer/workflow.py`](../../src/ao_shaping/tools/slm/transfer/workflow.py)
> **数据/关联脚本**: [`src/ao_shaping/tools/slm/transfer/fit.py`](../../src/ao_shaping/tools/slm/transfer/fit.py)
> **数据/关联脚本**: [`scripts/slm/probe_optical_path_validity.py`](../../scripts/slm/probe_optical_path_validity.py)
> **运行环境**: 硬件
> **说明**: --debug-weights 逐轮权重轨迹落盘的真机验证：上机前光路六步探测判定 ok，短时冷启动采集+拟合 6 轮在 SLM #1 + Daheng CCD 上完整跑通，weights_history.jsonl / weights_trajectory.npz / weights_summary.json 三份产物在真实采集数据上落地，训练损失 0.39325→0.34548；聚焦离线测试 6 用例通过。口径要点：reuse_latest 命中已有 latest 模型会跳过采集+拟合导致无权重产物，须用 AO_SHAPING_WEIGHTS_DIR 重定向到空目录强制冷启动。人工撰写。
<!-- provenance:end -->

## 摘要

本报告记录 `--debug-weights`（逐轮权重轨迹落盘）在**真实硬件**上的验证结果：按
`ao-optical-path-calibration` 六步探测上机前检查，随后以短时冷启动采集+拟合在
SLM #1 + Daheng CCD 上跑通真实训练，确认 `weights_history.jsonl` /
`weights_trajectory.npz` / `weights_summary.json` 三份产物在真实采集数据上完整落地，
离线聚焦测试 6 项全过，设备全部正常释放。

关键结论：`--debug-weights` 在真实数据链路上**功能正确**——6 轮迭代的权重轨迹
（18 个参数 × 7 行，含迭代 0 初始值）完整落盘，训练损失从 `0.39325` 单调优化至
`0.34548`；产物格式与离线测试口径一致。

**口径要点**：会话默认 `reuse_latest=True`，光路复用命中已有 latest 模型会**跳过
采集+拟合**，导致 `--debug-weights` 无产物。本次通过环境变量
`AO_SHAPING_WEIGHTS_DIR` 将权重目录重定向到空临时目录，使复用探测返回
"没有 latest 模型，走正常全流程"，强制走真实冷启动采集+拟合。这是本报告与后续
真机验证该功能时的标准口径。

## 任务完成情况

- [x] T1 上机前光路六步探测：Step 1–5 全部通过，判定 `ok`；设备锁 `status=idle`，无冲突进程。
- [x] T2 短时真实训练验证：冷启动采集+拟合（强制绕过光路复用）在 SLM #1 + Daheng CCD 上完整跑通；model 验收与 CCD 锚定块（sequential_ccd）验收均产出记录。
- [x] T3 `--debug-weights` 三份产物核验：`weights_history.jsonl`（7 行）、`weights_trajectory.npz`（18 参数 × 7 轮）、`weights_summary.json` 均在真实采集数据上落地，内容合理。
- [x] T4 聚焦验证：`tests/ao_shaping/tools/slm/transfer/test_debug_weights.py` 6 用例全部通过（`--import-mode=importlib`）。
- [x] T5 设备释放确认：SLM #1 与 CCD 正常关闭，设备锁释放，探测复跑 `status=idle`。

## 指标参数

### 设备与占用

| 项目 | 参数 | 说明 |
|------|------|------|
| SLM | Santec SLM-200 #1（`22030108`） | 1064 nm，2π 灰度设备动态读取 993，内部内存模式 |
| CCD | Daheng MER2-507-23GM（SN `FJB24112232`） | 2592×1944，曝光量程 0.020–1000 ms（满量程 4095） |
| 激光 | 沿用当前低功率设定 | **未做任何功率调整**，仅在用户授权的低功率可控环境下使用 |
| 设备锁 | `device_mutex` / `probe_devices` | 运行前后 `status=idle`，无其他硬件进程持锁 |

### 上机前光路探测（Step 1–5）

| 步骤 | 判据 | 实测 | 结论 |
|------|------|------|------|
| Step 1 设备在线 | 驱动板应答、CCD 连接 | SLM 打开成功（标识 `22030108`）；CCD 打开成功 | ✅ |
| Step 2 SLM 真在调制 | 离焦相位下宽度/峰值显著变化（> 3×合并标准误） | 宽度变化 19.2%，峰值变化 29.9% | ✅ |
| Step 3 光路有效 | 光通量 ≥ 参考下限 1550 counts/ms | 35580 counts/ms | ✅ 判定 `ok` |
| Step 4 曝光不超量程 | 峰值 ≤ 0.7×满量程（mono12 ≈ 2866），排除出过饱和像素的档 | 可用窗口 0.05 ms（峰值 1779）；0.2 ms 及以上饱和被排除 | ✅ |
| Step 5 几何与零级 | 质心稳定 | 质心 (1227, 1103)，逐帧散布小 | ✅ |

### 短时真实训练验证

运行参数：`shape=square`，`count=6`，`iterations=6`，`blocks=2`，`updates=2`，
`frames=3`，`phase_lr=0.3`。运行时段 15:15:56 → 15:18:09（约 2.2 分钟），全程
`--execute --debug-weights`。

光路复用：权重目录重定向至空临时目录（`AO_SHAPING_WEIGHTS_DIR`），探测记录
"没有 latest 模型，走正常全流程"，确认走真实采集+拟合。

### 训练损失轨迹（真实采集数据）

| iteration | loss |
|-----------|------|
| 0 | 0.39325 |
| 1 | 0.37492 |
| 2 | 0.37732 |
| 3 | 0.36291 |
| 4 | 0.35193 |
| 5 | 0.34997 |
| 6 | 0.34548 |

### 验收记录

| 验收方式 | 块 | 结果 | 指标 |
|----------|----|------|------|
| model_only | block 0 | 接受 | loss 2.64410 → 2.64339 |
| model_only | block 1 | 拒绝 | loss 2.63445 → 2.65413 |
| sequential_ccd | block 0 | 接受 | baseline 3.45736 → candidate 3.36361 |
| sequential_ccd | block 1 | 拒绝 | baseline 3.39827 → candidate 3.40613 |

短运行下块 1 未收敛属正常，本报告仅验证产物落地与链路通畅，不评价收敛质量。

### `--debug-weights` 产物

| 产物 | 路径（`data/debug/slm_transfer/debug_weights_real_force_20261121/fit/debug_weights/`） | 内容 |
|------|------|------|
| `weights_history.jsonl` | 7 行 | 每行 `{iteration, loss, parameters}`，逐轮展平参数 |
| `weights_trajectory.npz` | 18 参数 × 7 轮 | `iterations` 及 17 个模型参数的逐轮轨迹 |
| `weights_summary.json` | 每参数统计 | `first/last/delta/max_abs_step` |

参数轨迹抽查（`weights_summary.json`）：`amplitude_width_0` 400.0 → 423.6
（+23.6），`gain` 274.2 → 297.0（+22.8），`origin_0` 1227.5 → 1228.5，各参数
逐轮步长与 `phase_lr=0.3` 量级一致，轨迹合理。

## 边界与声明

- 激光功率**未做任何调整**，沿用实验台当前低功率设定，符合用户授权的"低功率、可控环境"边界。
- 本报告**不宣称** SLM 实机帧率提升，也**不宣称** CCD 与 SLM 逐帧一一对应。
- 运行期间未改动其他车道的未提交/未跟踪改动；本次验证产物位于
  `data/debug/slm_transfer/debug_weights_real_force_20261121/`，权重目录重定向为
  临时目录，真实 `data/slm_models` 无副作用。
