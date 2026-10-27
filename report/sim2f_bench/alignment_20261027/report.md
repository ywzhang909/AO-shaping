# 2f 实验系统孪生-硬件对齐结论 (Gate 0 归因汇总)

<!-- provenance:start -->
> **生成脚本**: 无（本报告汇总 lane 1 / lane 2 / optical_scale / session_hw 四路证据，各 lane 已有独立报告与脚本）
> **复现命令**: 见各 lane 报告
> **数据/关联脚本**: [`runs/sim2f_bench/README.md`](../../../runs/sim2f_bench/README.md) (预注册协议)
> **数据/关联脚本**: [`src/ml/sim2f/bench.py`](../../../src/ml/sim2f/bench.py) (run_session / Gate-0)
> **运行环境**: 离线汇总（只读 runs/sim2f_bench/session_hw 与各 lane 报告产物，无 SLM / 无相机）
> **说明**: 本报告把 lane 1 (仿真 dry run)、lane 2 (pupil_center_sweep 结论 A)、optical_scale (H1/H2 均否决) 与 session_hw (G0-2 FAIL / G0-4 FAIL) 四路证据汇总为一条对齐结论: G0-4 = 孪生保真度上限 (非放置错误), G0-2 = 真实光学漂移; 不放宽任何阈值。
<!-- provenance:end -->

- 被测会话: `runs/sim2f_bench/session_hw/` (实验系统, `--backend hw --samples loop`, 24 样本, `--record`)
- 校准参数: pupil（瞳面） center (960, 720), exposure 13.7846 ms (ceiling 18.8846 ms), pupil radius 450 px
- 写于: 2026-10-27

## 0. 预注册判据 (runs/sim2f_bench/README.md §4, 本报告未改)

| 门 | 阈值 | 含义 |
|---|---|---|
| G0-1 曝光 | flat peak ∈ [0.2, 0.8] of full scale, saturated < 0.3% | 相机曝光在有效线性区 |
| G0-2 零级稳定 | A1-vs-A2 0-order drift < 2 px (within sample) | 光路在两次测量间稳定 |
| G0-3 相位到达面板 | min norm(B-A)/norm(A1-A2) > 3 (24/24) | SLM 相位确实作用于光 |
| G0-4 几何可辨识 | min ncc(measured net, twin net) ≥ 0.5 | 孪生预测与实测几何对应 |

主判据: ncc_net > ncc_flat 的样本 ≥ 6 且 mean(ncc_net − ncc_flat) > 0.05。
G0-1~G0-4 任一未通过 ⇒ 主判据不报。

## 1. 四路证据汇总

### 1.1 Lane 1 — 仿真 dry run (sim backend)

- 入口: `run_bench.py --backend sim --samples loop --pupil-center-xy 960,720 --exposure-ms 13.7846`
- Gate 0: **全通过** (G0-1 N/A, G0-2 PASS, G0-3 PASS (inf), G0-4 PASS (1.000))
- 判据: 计入 23 (剔除 1), wins 5/6, mean delta −0.0907 < 0.05 ⇒ **net_does_not_recover**
- 结论: 孪生内部 U-Net 不优于 flat (仿真侧基线)

### 1.2 Lane 2 — pupil_center_sweep (离线, 纯读 runs/sim2f_bench/session)

- 脚本: `scripts/ml/calibrate_sim2f_pupil_center.py`
- 被测: `runs/sim2f_bench/session/` (hw, 24 样本), min ncc = 0.4898 (差门限 0.0102)
- 扫描: 13×13 网格 ±120 px, 细化 5 px
- 结论: **结论 A** — |Δ*| ≤ 20 px (最优偏移 (0,+20) px, 细化 (+10,-15) px, 距离 18.0 px)
- 含义: 瞳面放置**没错**; 0.4898 是孪生保真度上限, 不改 0.5, 不重跑整场

### 1.3 Optical scale (离线, 纯读 data/debug/model_in_loop_hw_collect)

- 脚本: `scripts/ml/diagnose_sim2f_optical_scale.py`
- H1 (远场尺度): 25 个 region_span_px 扫描点, **无一点**同时满足宽度比 ∈ [0.9,1.1] 与中位 ncc ≥ 0.75 ⇒ **否决**
- H2 (离焦/零级): defocus_m 相对差 1.2% (< 20%), 中位 ncc 提升 +0.0002 (< 0.05), 中位 L1 1.3324 (> 0.5) ⇒ **否决**
- 含义: 现有证据不足以说「标定已失效」; **不改任何已入库标定量**, 不重训网络

### 1.4 Hardware bench (session_hw, `--record`)

- Gate 0:
  - G0-1 曝光: **PASS** (flat peak 0.471–0.518, saturated 0.000%, exposure 13.7846 ms explicit)
  - G0-2 零级稳定: **FAIL** (max A1-vs-A2 drift 12.99 px, limit 2 px, worst sample 46; session spread 75.00 px; shaped excursion 22.68 px)
  - G0-3 相位到达面板: **PASS** (min ratio 5.21 > 3, 24/24)
  - G0-4 几何可辨识: **FAIL** (min ncc 0.190, floor 0.5)
- 主判据: 不报 (Gate 0 未通过)
- 过程数据: `session_hw/process_data/` (96 行, 含 _img/_phase sidecar, ~96 × 4.4 MiB)

## 2. G0-2 诊断: 0 级漂移为真实漂移

从 `session_hw/process_data/` 的 _img sidecar 重新计算 A1-vs-A2 0-order centroid (argmax + 3×3 加权质心):

| 统计量 | 值 |
|---|---|
| max drift | 12.99 px (sample 46) |
| mean drift | 2.85 px |
| median drift | 1.41 px |
| samples with drift > 2 px | 10 / 24 |

worst 5:
- sample 46: A1=(126.0,132.0) → A2=(113.0,132.0), drift 12.99 px
- sample 55: A1=(126.0,138.0) → A2=(117.0,131.0), drift 11.39 px
- sample 23: A1=(119.0,138.0) → A2=(112.0,133.0), drift 8.60 px
- sample 60: A1=(119.0,131.0) → A2=(123.0,137.0), drift 7.20 px
- sample 47: A1=(132.0,132.0) → A2=(127.0,132.0), drift 4.99 px

判定: 漂移**不是**随机噪声 (median 1.41 px 已接近门限, mean 2.85 px 超门限 1.4×)。
session spread 75.00 px (不同样本间的 0 级位置变异) 说明光路存在系统性漂移 (热/振动/对准)，
而非单帧噪声。G0-2 的 within-sample 设计 (A1 vs A2 同相位两次) 正是为排除相位影响,
因此该漂移只能归因于光路/设备本身。

## 3. 对齐结论

### 3.1 G0-4 = 孪生保真度上限 (非放置错误)

依据 lane 2 结论 A (|Δ*| = 18.0 px ≤ 20 px) + lane 1 (仿真 Gate 0 全过, 孪生内部一致) +
optical_scale (H1/H2 均否决, 标定未失效):
- 瞳面放置正确 (calibration/latest.json: panel center (960,720), offset [0,120])
- 孪生的 0.4898 min ncc 是保真度上限, 不是放置偏移
- **不改 0.5 门限**, 不重跑整场会话, 不重训网络

session_hw 的 0.190 ncc (远低于 0.4898) 与 G0-2 的 12.99 px 漂移强相关:
0 级位置漂移导致远场测量与孪生 net 不可比, ncc 直接崩到 0.19。修好 G0-2 后 G0-4 应回到
~0.49 量级 (lane 2 的结论 A 范围)。

### 3.2 G0-2 = 真实光学漂移 (需修光路)

- 12.99 px 漂移 (6.5× 门限) 是真实信号, 不是噪声
- 可能原因: 热漂移 (session ~180 s)、SLM 未 settle、相机曝光时序、振动
- 建议: 加长 settle 时间, 检查 SLM memory slot 轮换, 物理稳定措施, 重跑验证可复现性
- **不放宽 2 px 门限**

### 3.3 仿真-硬件 delta 差异

| 指标 | 仿真 (lane 1) | 硬件 (session_hw) |
|---|---|---|
| mean delta (ncc_net − ncc_flat) | −0.0907 | −0.1989 |
| wins (ncc_net > ncc_flat) | 5/6 | 0/6 |
| Gate 0 | PASS | FAIL (G0-2, G0-4) |

硬件的 delta 更差 (-0.1989 vs -0.0907) 且 wins 0/6 (vs 5/6), 与 G0-2 漂移一致:
漂移使 net 帧与 twin 不可比, 优化后的 net 反而比 flat 差。

## 4. 不做的事 (红线)

- 不放宽 G0-2 的 2 px 门限
- 不放宽 G0-4 的 0.5 门限
- 不重跑整场会话 (结论 A, 放置正确)
- 不改已入库标定量 (optical_scale H1/H2 均否决)
- 不重训网络 (lane 1 基线已建立)
- 不在代码注释中写实验叙述 (写进本报告, 代码只留不变量 + 报告链接)

## 5. 下一步

1. 修光路 (G0-2 漂移): 加长 settle / 检查 SLM slot 轮换 / 物理稳定
2. 重跑硬件 bench 验证漂移可复现性
3. 若漂移修好, G0-4 应回到 ~0.49 量级 (lane 2 结论 A)
4. 若 G0-4 仍 < 0.5 且放置正确 → 孪生保真度上限, 需改进孪生模型 (非标定)
5. 提交范围确认 (本报告 + runs/sim2f_bench/run_bench.py 的 --record 修改)

## 6. 产物

| 文件 | 内容 |
|---|---|
| `runs/sim2f_bench/session_hw/VERDICT.md` | Gate 0 + 判据 + 结论 |
| `runs/sim2f_bench/session_hw/session.json` | 会话元数据 + rows |
| `runs/sim2f_bench/session_hw/rows.csv` | 24 样本 × 指标 |
| `runs/sim2f_bench/session_hw/frames.npz` | A1/B1/B2/A2 帧 |
| `runs/sim2f_bench/session_hw/process_data/` | BenchRecorder 产物 (96 行 + sidecar) |
| `report/sim2f_bench/twin_vs_hw_corpus/20261027/pupil_center_sweep.md` | lane 2 结论 A |
| `report/sim2f_bench/optical_scale_20261027/report.md` | optical_scale H1/H2 否决 |
| `report/sim2f_bench/alignment_20261027/report.md` | 本报告 |
