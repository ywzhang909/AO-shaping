# #213 墙钟成本分解 — repro1200_20261109 会话（2026-11-11 离线分析）

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`scripts/slm/analyze_session_time_budget.py`](../../../scripts/slm/analyze_session_time_budget.py)
> **数据/关联脚本**: [`src/ao_shaping/tools/kernels/bench_kernels.py`](../../../src/ao_shaping/tools/kernels/bench_kernels.py)
> **运行环境**: 离线
> **说明**: #213 墙钟成本分解：repro1200_20261109 会话（2808.2 s、594 帧）；瓶颈为 LCOS settle 等待（帧间 gap 1212 s，AB 双峰 5-12 s/次）与模型/seed 构建 （transport_cv 首帧前约 130 s/shape）；优化器 71 次迭代仅占 0.02-0.03%，GPU 提速 12.58->7.88 ms/iter 全会话省 0.33 s。输入为磁盘存档，未占用设备。
<!-- provenance:end -->

## 0. 结论摘要

对 issue #213（收敛效率 / 优化策略）的墙钟证据：

- **优化器不是瓶颈。** 整个 2808.2 s 会话中，shape 阶段共 **71 次**优化器迭代。GPU 单步 12.583 ms（head）→ 7.880 ms（worktree），全会话墙钟收益 **0.33 s（0.012%）**；优化器总占比 0.032% → 0.020%。`torch.compile` 类 2–3× 算子加速对墙钟收益 <1 s，P0 优先级不成立。
- **瓶颈 #1 是 LCOS 液晶稳定等待**：shape 阶段帧间 gap 合计 **1212.0 s**（采集+间隔总盘 1224.6 s 的 99.0%），其中 ≥0.6 s 的 171 个 gap 占 1200.7 s。AB 结构显示每次采集背 5–12 s 的 settle 税，与 `display_and_average` 的逐帧稳定性探测协议（`n_stable=3`、容差 2%、单次封顶 6.0 s）一致。
- **瓶颈 #2 是模型/seed/目标构建**：shape 阶段的 `stage_other` 约 677.0 s（35.0%），主体是 `transport_cv` 每 shape 首帧前 127–134 s 的一次性构建（×3 shape ≈ 390 s）与 `mono12_*` 每阶段首帧前 30–34 s。
- **曝光调节** 71.3 s（2.5%），前馈预估最多省约 35 s，P1 优先级应下调。
- **相机读出** 12.6–13.5 s（0.5–1.0%），不在瓶颈之列。

高杠杆方向因此反转：降低/并行化 settle 等待（放宽容差、减少稳定帧数、双缓冲交替显示）或减少迭代次数（少背 settle 税），远比优化器提速有价值。异步采集-计算流水线（P3）若不同时隐藏 settle，收益有限。

## 1. 数据源与方法

| 项 | 值 |
|---|---|
| 主会话 | `data/debug/slm_transfer/repro1200_20261109/`（完整，3 shape × 4 stage） |
| 交叉验证 | `data/debug/slm_transfer/multishape1200_20261027/`（部分完成，1585 s，结构同构） |
| GPU 基准 | `data/debug/slm_transfer/diffopt_dispatch_20261110/results_{head,worktree}.json` |
| 分析脚本 | `scripts/slm/analyze_session_time_budget.py`（既有，未改动） |
| 硬件基准环境 | RTX 5080 Laptop GPU，30 iter + 10 warmup，冻结 checkpoint（`repro1200_aperture_closed_20261109_1910/fit`），每迭代显式 synchronize 计时 |

分析方法：journal.json 的 `running`→`complete` 配对给阶段窗口；`shape/<shape>/<recipe>/manifest.json` 每条采集记录的 `frames[]` 行含 `monotonic_ns`/`end_monotonic_ns`/`exposure_ms`，给出相机曝光+读出时长与帧间 gap；两者之差为 `stage_other`。本报告全部数字来自磁盘存档，未占用任何设备（未打开 SLM/相机/激光，未 import `ao_shaping.drivers`）。

复现命令（纯离线）：

```powershell
.\.venv\Scripts\python.exe scripts/slm/analyze_session_time_budget.py --session data/debug/slm_transfer/repro1200_20261109
```

**独立复核**：主会话数字由第二套临时脚本（不信任分析脚本输出、直接扫 manifest/journal）重算确认，一致。两处口径差异已核对：分析脚本表中的 594 帧 = 528 shape 帧 + 66 capture 阶段帧；独立扫描的 546 shape 帧多出的 18 帧来自被中断的 `star/mono12_continue`（分析脚本按完成阶段计，未含）。

## 2. 会话总览

会话总时长 **2808.2 s**。journal 配对出 13 个完整阶段窗口（合计 2281.7 s）；`capture_geometry` 完成（08:24:55）到首个 shape 阶段启动（08:33:41）之间的 526.5 s 为 `seed_bakeoff` 窗口。

| 阶段 | 窗口 (s) | 相机 (s) | gap (s) | 曝光调节 (s) | 其他 (s) | 帧数 |
|---|---|---|---|---|---|---|
| capture | 173.7 | 1.4 | 158.2 | 0.0 | 14.1 | 66 |
| fit | 34.3 | 0.0 | 0.0 | 0.0 | 34.3 | 0 |
| capture_geometry | 140.3 | 0.0 | 0.0 | 0.0 | 140.3 | 0 |
| square/transport_cv | 181.7 | 0.3 | 46.0 | 6.4 | 129.0 | 36 |
| square/mono12_cv | 190.1 | 1.5 | 143.5 | 7.1 | 38.0 | 60 |
| square/mono12_continue | 264.9 | 2.3 | 211.0 | 7.1 | 44.6 | 84 |
| square/mono12_dense | 104.4 | 0.9 | 62.0 | 6.9 | 34.5 | 36 |
| triangle/transport_cv | 192.8 | 0.3 | 49.4 | 7.6 | 135.6 | 36 |
| triangle/mono12_cv | 209.6 | 1.5 | 157.0 | 8.1 | 42.9 | 60 |
| triangle/mono12_continue | 282.1 | 2.4 | 225.3 | 7.6 | 46.9 | 84 |
| triangle/mono12_dense | 124.5 | 0.9 | 79.3 | 7.0 | 37.4 | 36 |
| star/transport_cv | 178.4 | 0.3 | 43.4 | 6.6 | 128.0 | 36 |
| star/mono12_cv | 204.9 | 1.8 | 156.0 | 7.0 | 40.1 | 60 |
| **合计（配对阶段）** | **2281.7** | **13.5** | **1331.2** | **71.3** | **865.7** | **594** |

全会话占比：相机 0.5%、settle/其他 gap 47.4%、曝光调节 2.5%、阶段开销 30.8%。每采集帧平均 0.02 s 相机 + 2.24 s gap。

## 3. 瓶颈 #1：LCOS 液晶稳定等待（帧间 gap）

shape 阶段 manifest 独立复算：**546 帧，相机 12.6 s（1.0%），帧间 gap 1212.0 s（99.0%）**。gap 呈双峰：

| gap 档 | 数量 | 合计 (s) |
|---|---|---|
| ≥0.6 s（迭代间/settle） | 171 | 1200.7 |
| <0.6 s（ABBA 帧内相机流水线） | 364 | 11.2 |

`square/mono12_continue` 的逐采集 gap（A1/B 变体交替）直接显示结构：

| 变体 | gap 范围 (s) | 含义 |
|---|---|---|
| A1（B→A1，大图案切换） | 11.77–12.17 | 新迭代图案写入 + 大灰度变化的长 settle |
| B（A1→B，小变化） | 4.87–5.65 | 同迭代内参考变体，灰度变化小，settle 短 |

机理对应 `src/ao_shaping/tools/kernels/bench_kernels.py:720-733` 的 `display_and_average`：写相位后至少等 `DEFAULT_MIN_WAIT_S=0.3 s`，随后逐帧读取直到连续 `DEFAULT_N_STABLE=3` 帧在峰值/位置/宽度上互差 <2%，单次封顶 `max_wait_s=6.0 s`。驱动侧的翻转时间估计随灰度图变化量增大，解释了 A1（~12 s）与 B（~5 s）的不对称。本会话 `config.json` 未覆盖这些参数，全部走默认值。

每次优化迭代以 ABBA 对采集 2 次，71 次迭代 ≈ 142 次采集 × 平均 ~8.5 s settle ≈ 1.2×10³ s —— 与实测 1212 s 自洽。**settle 税是按次征收的：减少迭代次数或放宽 settle 判据，收益直接线性。**

## 4. 瓶颈 #2：模型 / seed / 目标构建（stage_other）

shape 阶段 `stage_other` 合计约 **677.0 s**（shape 阶段 1933.4 s 的 35.0%）：

- `transport_cv` 阶段 journal 窗口 178–193 s，其中**首帧前 126.9–133.6 s** 是模型加载 + 目标构造 + seed 烘焙（square 126.9 s、triangle 133.6 s、star 126.6 s）。三个 shape 一次性构建合计约 390 s。
- `mono12_*` 每阶段首帧前 30.0–34.4 s，8 个阶段合计约 250 s。

这部分是 CPU 侧的构建工作，与 GPU 单步速度无关；跨阶段复用构建产物（同模型不同 shape 的公共部分）是直接候选。

## 5. 非瓶颈：优化器、相机、曝光调节

**优化器。** shape 阶段共 71 次迭代（按 manifest 中 `_<idx>_A1/B` 标签的 distinct idx 计数）。GPU 基准（`diffopt_dispatch_20261110`）：

| 版本 | wall median (ms/iter) | p10 | p90 |
|---|---|---|---|
| head | 12.583 | 10.957 | 14.302 |
| worktree | 7.880 | 7.666 | 8.984 |

- 71 次迭代总优化器时间：head 0.89 s → worktree 0.56 s，占会话 0.032% → 0.020%。
- 本次 GPU 提速的全会话墙钟收益：**0.33 s（0.012%）**。即便 3× 加速，收益 <0.6 s。

**相机。** 曝光+读出+传输合计 12.6–13.5 s（0.5–1.0%），每帧约 23 ms（含 25.92 ms 曝光档）。

**曝光调节。** 反馈式 `tune_exposure` 每阶段 6.4–8.1 s，合计 71.3 s（2.5%）。前馈预估（P1）按省 50% 计约 35 s，绝对收益小。

## 6. 对 #213 各子项的含义

| #213 子项 | 数据含义 | 建议 |
|---|---|---|
| P0 算子融合 / torch.compile 2–3× | 优化器占 0.02–0.03%，收益 <1 s | **降为不做或最后做** |
| P1 曝光前馈预估 | 基数 71.3 s，最多省 ~35 s | 优先级下调 |
| P3 异步采集-计算流水线 | 60% 是液晶稳定等待而非传输 I/O；不隐藏 settle 则收益有限 | 保留但改目标：必须重叠 settle 与计算 |
| —（新增候选）放宽/并行化 settle | 每迭代 ~2×8.5 s settle 税；`settle_tol`、`n_stable`、双缓冲交替显示直接减税 | **提为最高优先** |
| —（新增候选）减少迭代次数 | settle 税按次征收，多尺度/分层优化减少迭代即线性省时 | 与 settle 项并列 |
| —（新增候选）复用模型/seed 构建 | transport_cv 构建 ~130 s × 3 shape | 中优先 |

## 7. 无法离线归因的部分

manifest 帧行仅有 4 个键（`timestamp_utc`/`monotonic_ns`/`end_monotonic_ns`/`exposure_ms`），**无** `slm_write`/`settle`/`read`/`reduce`/`save` 分段计时。因此：

- 帧间 gap 内部（SLM 写入 vs 液晶稳定 vs 落盘 vs 优化器）只能按差额归因，不能精确拆分。需下次实验以 `--profile-timing` 重跑（runner 已有该 flag，会在 manifest 每行记录分段 ms）。
- 落盘代价（464 个 npz ≈ 3.6 GB）含在 gap/other 内；按 14.2 MB/帧、SSD 写入 <0.1 s/帧估计，对 5 s 级 gap 影响 <2%，但不是实测。
- `seed_bakeoff` 窗口（526.5 s）与 `capture`/`capture_geometry` 的内部分解不可离线完成。

## 8. 局限

- 单会话样本（2026-11-09 `repro1200_20261109`，同一 2f 光路、同一模型）；`multishape1200_20261027` 会话结构同构但被中断，仅作定性交叉验证。未做跨日重复。
- 占比分母：§3–§5 以 shape 阶段 1933.4 s 或会话 2808.2 s 为分母，已在各处标明。
- GPU 基准为纯 GPU 中位（显式 synchronize）；实际每步墙钟还受 CPU 包络限制，CPU 侧包络为单测值（44.49→26.46 ms/步），未在目标机同时测 CPU+CUDA 墙钟。
