# 光阑关闭状态 side=1200 整形复现与可微优化效率（2026-11-09）

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`scripts/slm/diffopt_efficiency_probe_20261108.py`](../../../scripts/slm/diffopt_efficiency_probe_20261108.py)
> **数据/关联脚本**: [`scripts/slm/diffopt_dispatch_probe_20261110.py`](../../../scripts/slm/diffopt_dispatch_probe_20261110.py)
> **数据/关联脚本**: [`src/ao_shaping/runners/slm/transfer_shapes_runner.py`](../../../src/ao_shaping/runners/slm/transfer_shapes_runner.py)
> **数据/关联脚本**: [`src/ao_shaping/tools/slm/transfer/replay_phases.py`](../../../src/ao_shaping/tools/slm/transfer/replay_phases.py)
> **数据/关联脚本**: [`src/ao_shaping/algorithm/signal_processing/differentiable_defocus_uniformity.py`](../../../src/ao_shaping/algorithm/signal_processing/differentiable_defocus_uniformity.py)
> **运行环境**: 实验+离线
> **说明**: 光阑关闭后光路(零级未压制 E0/E1≈1.2, 曝光窗 0.05-0.2ms, 几何未变)重做 side=1200 square/triangle/star 整形: 全新标定 campaign 复现 + 共同重放成功(0.05/0.2ms) + 可微优化 GPU/学习率效率拆解; 含与 10-28 基线同口径对比(跨曝光不作直接对等)
<!-- provenance:end -->

> **一句话**：光阑关闭态下 side=1200 整形**未复现** 10-28 收敛——mono12 各块 Jacobian 全部 `contradict`（在线方向不可信），末帧 CV 无实质改善（1.07–1.17），共同重放在唯一可信曝光（0.2 ms，core_peak 364–420）下 CV 0.54–0.69、target_pearson ≤0.22，**无成形结构**。主导假设是**未压制零级（E0/E1≈1.2）污染目标域**；`locate()` 零级掩膜只修探测不修物理零级。效率侧：GPU 全链路 47–90 ms/步 vs CPU 630 ms/步（~7–13×），但 torch.profiler 显示 GPU 计算仅 ~4.6 ms/迭代、**~19 ms 为 Python 逐算子调度**（~400 算子/迭代，`aten::roll` 24×/迭代），省时瓶颈在调度而非 GPU 吞吐；lr 0.01 已近收敛地板（离线 loss 9.33e-6，0.05 仅 9.05e-6）。**2026-11-10 调度减负落地**：热机 GPU 每步 12.58→**7.88 ms**（−37%）、CPU 44.49→**26.46 ms**（−40%），同步 60→13 次/迭代、`aten::nonzero` 8→0，**逐位行为中性**；原「合并 roll+FFT / `fft256`」方向**已证伪**（FFT+roll 仅 ~0.1 ms/迭代）。

---

## 1. 实验设定与方法

- **光路状态**：光阑关闭后（前置探针 `verdict=ok`、exit=0，`data/debug/optical_path_validity/aperture_closed_retry_20261109_20261109_190027/20261109_110034Z/optical_path_probe.json`）。可用曝光窗 **0.05–0.2 ms**（10-28 为 1–10 ms，强度约基线 7.5×）；离焦响应显著；SLM 正常关闭。零级未压制（E0/E1≈1.2），几何映射未变（见 [optical_path_status_20261109.md](../optical_path_status_20261109.md) §2）。
- **campaign**：`data/debug/slm_transfer/repro1200_aperture_closed_20261109_1910/`（独立输出目录、全新标定、`--no-auto-reuse`，不复用光阑关闭前 campaign）。
- **参数**：period=48、window_span=760、window_step=4、zero_radius=60、capture_grid=384、capture_count=12、fit_iterations=150、control_grid=640、fft_size=2048、initial_phase=auto、seed_bakeoff、gs_seed、jacobian_guard、`--no-tilt`、objective=cv、weight_schedule=constant。
- **设备纪律**：SLM #1 + CCD 仅 replay 占用；replay exit=0，`common_replay/released.json` 记录 `"camera and SLM contexts closed"`（12:29:51.384509Z），SLM #1 正常关闭。
- **阶段**：plan → capture（183.9 s）→ fit（38.2 s）→ geometry → model → capture_geometry（145.9 s）→ wide_model → seed_bakeoff×3 → 12 形状阶段（每形状 transport_cv / mono12_cv / mono12_continue / mono12_dense）→ replay。
- **10-28 基线**：`data/debug/slm_transfer/multishape_scale_20261028/s1200/`（同流程，曝光 20/50/90 ms，强度低 7.5×）。**跨曝光、跨强度的 counts/CV 不可直接对等**（同曝光重放才是可比口径，见 [multishape1200_20261027.md](../multishape1200/multishape1200_20261027.md) §1.2）。

## 2. 物理预测模型拟合（fit）

150 迭代，`device=cuda`（`device_requested=auto`），`record_responses=false`：

| 项 | 本轮（光阑关闭态） | 10-28 基线 | 说明 |
|----|------------------|-----------|------|
| loss 0→150 | 0.778 → **0.233** | 1.087 → 0.398 | 本轮终值更低 |
| 墙钟（首 mono → 末 end） | **19.96 s** | 146.52 s | ~7.3× 快（GPU） |
| 每迭代均值相关（0→150） | 0.483 → **0.785** | 0.272 → 0.681 | 本轮更高 |
| 最终相关（12 透镜，0–1） | 0.568–0.967 | 0.384–0.918 | — |
| defocus_diopter | 0.262 | 0.330 | — |
| origin (px) | (1101.9, 1164.7) | (1338.0, 949.4) | 零点贴近底边 |
| gain | 4169.8 | 69.2 | ~60×（强度更高） |
| background (counts) | 45.28 | 0.639 | — |

**注意**：本轮 `record_responses=false`，`result.json` 无 `responses` 数组（10-28 有）；correlation 仅存于 `metrics[i].correlation`，无顶层 `corr`。拟合本身健康（相关 0.57–0.97），但**拟合的是含零级污染的物理预测模型**，未对零级做物理扣除。

## 3. 整形阶段（12 个形状阶段，实测闭环）

每形状四段：transport_cv（8-bit）→ mono12_cv → mono12_continue → mono12_dense（window_step 2.0）。段末帧指标（mono12_dense 末帧 = 最终相位）：

| 形状 | 阶段 | 块 accept | CV 起点→末帧（mono12_dense 末） | target_pearson 起点→末帧 | core_peak 末帧 | Jacobian |
|------|------|-----------|-------------------------------|--------------------------|----------------|----------|
| square | mono12_cv | 3/8 | 1.124 → **1.167** | 0.093 → 0.093 | 1432 | `contradict` 8/8 |
| square | mono12_continue | 7/12 | 1.160 → **1.186** | 0.092 → 0.092 | 1364 | `contradict` 12/12 |
| square | mono12_dense | 2/4 | 1.197 → **1.167** | 0.092 → 0.093 | 1310 | `contradict` 4/4 |
| triangle | mono12_cv | 5/8 | 1.109 → 1.097 | 0.228 → 0.224 | 1273 | `contradict` 8/8 |
| triangle | mono12_continue | 5/12 | 1.120 → 1.084 | 0.224 → 0.226 | 1186 | `contradict` 12/12 |
| triangle | mono12_dense | 1/4 | 1.080 → **1.071** | 0.228 → 0.230 | 1194 | `contradict` 4/4 |
| star | mono12_cv | 5/8 | 1.131 → 1.128 | 0.228 → 0.228 | 1251 | `contradict` 8/8 |
| star | mono12_continue | 7/12 | 1.140 → 1.100 | 0.227 → 0.228 | 1237 | `contradict` 12/12 |
| star | mono12_dense | 3/4 | 1.098 → **1.117** | 0.227 → 0.232 | 1236 | `contradict` 4/4 |

（CV 起点 = mono12_cv 首帧；target_pearson 起点 = mono12_cv 首帧。transport_cv 段 CV 1.22–1.49，无 Jacobian key。）

**关键观察**：

1. **CV 无实质改善**：末帧 CV 1.07–1.19，起点 1.08–1.20，全程在 1.05–1.20 窄带内抖动——没有收敛趋势。10-28 基线同阶段 CV 0.65–0.73（mono12_cv 末 0.66–0.71），**本轮 CV 口径整体偏高 ~1.6×**。
2. **Jacobian 全部 `contradict`**：square/triangle/star 各 mono12_cv（8 块）+ continue（12 块）+ dense（4 块）= **72 块全部 `contradict`**（预测残差与实测方向不一致）；transport_cv 无 Jacobian key（`None` verdict）。→ **在线优化方向不可信**，闭环无法改善是结构性的，非起点问题。
3. **core 未饱和**：mono12 段 core_peak 1091–1515（≈27–37% full_scale=4095），target_peak 恒 4095 但饱和 DC 在零级盘内，core 未饱和（曝光审计：12-bit full_scale=4095 正确，饱和不在 core）。
4. **零级污染**：`config.json` 中 `zero_xy == target_xy == (1096,1175)`（381×381 采样网格中心，疑默认值），但原始帧亮斑 top-0.1% 质心 ≈ (1095,1166)，偏差 ~9 px；`zero_radius=60` 只从评价里排除零级盘，**没有物理消除零级**。

## 4. 共同重放（replay，最终帧可比口径）

重放成功（exit=0，`common_replay/released.json` 已写）：30 条目 × 3 帧 = 90 帧，15 标签，曝光 **0.05/0.2 ms**。旧失败目录改名 `common_replay_aborted_50ms_20261109/`（曝光 50/100 ms 首条 `core_peak=4095` 全饱和 → 饱和无法测光，故改用 0.05/0.2 ms）。

### 4.1 @0.05 ms（近噪声下限，不作成形证据）

| 标签 | core_peak | CV | target_pearson | target_peak | mean |
|------|-----------|-----|----------------|-------------|------|
| transport_cv_square_seed | 140.7 | 0.205 | 0.093 | 418.0 | 16.63 |
| transport_cv_triangle_seed | 134.7 | 0.335 | 0.221 | 489.0 | 19.48 |
| transport_cv_star_seed | 134.7 | 0.345 | 0.216 | 536.3 | 19.54 |

（全部 mono12_final @0.05 ms：square CV 0.217–0.221，triangle 0.317–0.329，star 0.323–0.334，core_peak 127–150，mean 16–19 counts。）

**@0.05 ms 的 core_peak 仅 127–150（≈3% full_scale）、mean 16–19 counts**——信号近噪声下限，CV 0.20–0.34 是**近空图低相对方差伪影**，不是成形证据。

### 4.2 @0.2 ms（可信曝光，最终指标）

| 标签 | core_peak | CV | target_pearson | target_peak | mean | ee_roi |
|------|-----------|-----|----------------|-------------|------|--------|
| transport_cv_square_seed | 367.3 | 0.548 | 0.088 | 1597.0 | 19.95 | 0.711 |
| mono12_dense_square_final | 389.7 | 0.544 | 0.088 | 1601.0 | 19.93 | 0.711 |
| transport_cv_triangle_seed | 414.0 | 0.709 | 0.218 | 1524.0 | 28.37 | 0.374 |
| mono12_dense_triangle_final | 385.3 | 0.687 | 0.218 | 1543.0 | 28.30 | 0.375 |
| transport_cv_star_seed | 420.7 | 0.735 | 0.219 | 1564.3 | 28.76 | 0.364 |
| mono12_dense_star_final | 397.7 | 0.693 | 0.218 | 1556.0 | 28.48 | 0.363 |

（@0.2 ms 全标签：square CV 0.543–0.548，triangle 0.676–0.687，star 0.691–0.735；core_peak 364–420 ≈ 9–10% full_scale，mean 19–28 counts。）

**@0.2 ms 是可信口径**：core_peak 364–420（10% full_scale），CV square 0.54 / triangle 0.68 / star 0.69，target_pearson square 0.088 / triangle 0.218 / star 0.218——**target_pearson ≤0.22 表明目标域与形状掩膜无相关，无成形结构**。

### 4.3 与 10-28 基线对比（跨曝光，方向性参考）

10-28 `common_replay` @20 ms：square 0.419 / triangle 0.491 / star 0.572（@50 ms 0.596/0.618/0.677）。**本轮 @0.2 ms 与 10-28 @20 ms 曝光差 100×、强度差 7.5×，不可直接对等**；仅能说 10-28 在更低信号下 CV 更低，方向性提示本轮更差，但**不是同口径结论**。10-28 无 `target_pearson`（未记录），无法做该口径对比。

## 5. 可微优化效率（GPU + 学习率）

### 5.1 离线基准（冻结 checkpoint，`ShapeObjective(cv_only=True)` + `gs_seed` 同路径）

**每次 `update()` 中位耗时**（同一脚本、同一输入，两次运行；`ms_p90` 同列后附）：

| 设备 | lr | 迭代 | 首次冷跑 (ms) | 复跑热机 (ms) | loss 初→终 |
|------|-----|------|--------------|--------------|-----------|
| GPU (RTX 5080 Laptop) | 0.01 | 20 | 47.2 (p90 51.9) | **21.2** (p90 35.7) | — |
| CPU | 0.01 | 20 | 630.3 (p90 1751.5) | **49.2** (p90 57.6) | — |
| GPU | 0.001 | 150 | 49.0 | **12.3** | 8.68e-5 → 1.88e-5 |
| GPU | 0.003 | 150 | 65.4 | **11.6** | 8.68e-5 → 9.60e-6 |
| GPU | 0.01 | 150 | 52.5 | **12.7** | 8.68e-5 → 9.33e-6 |
| GPU | 0.02 | 150 | 85.1 | **12.6** | 8.68e-5 → 9.24e-6 |
| GPU | 0.05 | 150 | 90.0 | **11.9** | 8.68e-5 → 9.05e-6 |

- **GPU 快于 CPU**：热机复跑 20 迭代 **21.2 vs 49.2 ms（2.3×）**；首次冷跑 47.2 vs 630.3 ms（13.3×）。**两次数值差主要由 GPU 时钟/预热状态主导**（同代码同输入 loss 逐位一致，见 §5.3），故"GPU 加速比"应报 **2.3×（热机，保守）到 13×（冷跑）**，不能用冷跑 13× 当作稳定倍率。
- **lr 收敛（两次一致，可靠）**：0.001→0.05 终值 1.88e-5 → 9.05e-6；**0.01 已 9.33e-6，接近地板**，0.02/0.05 相对 0.01 仅 −1%/−3%（边际）。收敛速度上 0.02/0.05 更快到地板（更快压缩迭代）。
- ⚠️ **口径**：离线 loss ~9e-6 与硬件 CV（~1.1）不同口径，**不可对等**；仅用于耗时与 lr 收敛趋势。

### 5.2 torch.profiler（`update()` 每步成本拆解）

| 项 | 值 | 说明 |
|----|-----|------|
| GPU 实际计算 | **~4.6 ms/迭代** | 反传+前向 kernel |
| Python 逐算子调度 | **~19 ms/迭代** | ~400 个 torch 算子/迭代；`aten::roll` 24 次/迭代，占 self CUDA 8.7% |
| 同步开销 | ~0.08 ms | 可忽略 |

→ **省时瓶颈在算子调度，不在 GPU 吞吐**（定性；调度 ≈ 数倍 GPU 计算）。⚠️ torch.profiler 逐算子插桩会**抬高绝对耗时**（本剖析 ~24 ms/迭代 vs §5.1 热机稳态 12–21 ms 墙钟），故 §5.1 墙钟为准、本节的**相对拆解**（调度主导）为结论。

⚠️ **本节原先给出的下一步方向「合并 roll+FFT / 粗网格 `fft256`」已被 2026-11-10 的复测证伪**（见 §5.2.1）：一次 `update()` 里 FFT 与 `aten::roll` 的 self CUDA 合计只有 **~0.1 ms/迭代**，省时点在**每次迭代的主机同步次数与算子发射次数**，不在 FFT 吞吐。正确方向是消除同步与重复建张量，不是压 FFT。

### 5.2.1 2026-11-10 复测与 A/B：调度瓶颈的定位与处置

**定位**：逐算子拆解（`scripts/slm/diffopt_dispatch_probe_20261110.py`，冻结 checkpoint、离线、无设备占用）显示 FFT 与 `aten::roll` 合计仅 **~0.1 ms/迭代** → 「合并 roll+FFT」最多省 0.1 ms，**方向无效**。真正的成本是每次迭代的**主机同步**（`cudaStreamSynchronize` **60 次/迭代**）与**算子发射**（`cudaLaunchKernel` **731 次/迭代**）：其中 8 次 `aten::nonzero`（布尔掩膜索引内部的主机同步）与 7 次逐个 `float()`（诊断标量回读）都来自 `ShapeObjective.evaluate`。

**A/B（同一脚本、同一 checkpoint、同一 GPU；改前 = `git archive HEAD src` 取出的源码树，改后 = 本轮工作树）**：

| 指标 | 改前 | 改后 | 变化 |
|------|------|------|------|
| 每步墙钟中位（GPU，热机） | 12.58 ms | **7.88 ms** | **−37%** |
| 每步墙钟中位（CPU） | 44.49 ms | **26.46 ms** | **−40%** |
| `cudaLaunchKernel` | 731/迭代 | 499/迭代 | −232 |
| `cudaStreamSynchronize` | 60/迭代 | **13/迭代** | −47 |
| `cudaMemcpyAsync` | 76/迭代 | 29/迭代 | −47 |
| `aten::nonzero` | 8/迭代 | **0** | −8（该主机同步源全部消除） |
| 算子总数 | 4445/迭代 | 3044/迭代 | −1401 |

产物：`data/debug/slm_transfer/diffopt_dispatch_20261110/results_{head,worktree}.json`。复现：`git archive --format=tar --output=head.tar HEAD src && tar -xf head.tar -C <dir>`，再 `python scripts/slm/diffopt_dispatch_probe_20261110.py --src <dir>/src --label head`（改后不传 `--src`）。

**处置**：① `evaluate` 的掩膜索引改扁平索引（免 `nonzero` 主机同步）；② 7 个诊断标量合并成一次 `torch.stack([...]).tolist()`（7 次同步 → 1 次）；③ 前向的相机采样网格在 `__init__` 预计算一次（`predict()` 不再每次重算）。代码与逐位证据见 §5.4。

### 5.3 2026-11-09 代码优化落地（已提交 `d92faac2`）

- **anchored 前向复用**（`differentiable_defocus_uniformity.py::update()`）：`_accepted` 在单次 `update()` 内不变，复用 `seed + accepted` 前向结果（`anchor_cache`）替代第 4 次前向——**每步 4→3 次前向**。顺带修复 squeeze 形状回归（两个预测各自独立 `squeeze`，原先合并处理使 candidate 保持 `(1,8,8)` 触发 mask 索引错误）。`test_measured_phase_regularization.py` 2 通过、ruff 通过；`bench_optim.py` 复跑 loss **逐位一致**（9.326e-6 / 9.237e-6 / 9.046e-6，与改动前完全相同）→ **改动是行为中性**。
- **复跑耗时（优化后，热机）**：GPU 20 迭代 21.2 ms/步、150 迭代 11.6–12.7 ms/步；CPU 20 迭代 49.2 ms/步。**但**与首次冷跑（47–90 / 630 ms）的差值**不能归因于 4→3 前向**——loss 逐位一致证明行为未变，差值是 GPU 时钟/预热状态主导（冷跑 GPU 未升频）。**结论：本改动只减少每步前向次数（≤前向成本的 25%），不构成可测的稳定加速；热机稳态 GPU ~12–21 ms/步 才是可信基准。**
- **`validate.py` phase_lr 默认保持 0.01**（未改）：离线基准建议 0.02，但需硬件实验验证后再定。
- **GPU 路径**：campaign 已用 GPU（`auto`→cuda，provenance `device=cuda`），无额外代码改动；效率收益已由 §5.1 量化。
- `multishape_objective.py`：仅报错文案措辞修正（FFT 网格可达窗口消息），无语义变更。

### 5.4 2026-11-10 调度减负（行为中性，CI 门禁钉住）

代码改动两文件三处：

- **掩膜索引缓存**（`multishape_objective.py`）：`image[bool_mask]`（内部 `nonzero` → 主机同步）换成 `image.reshape(-1)[idx]`；`idx` 由 `np.flatnonzero` 在 **CPU 侧**算好，按**内容**缓存（`np.array_equal` 比对）——外部原地改 `core`/`box` 会自动失效，**不会返回过期掩膜**。每个 gather 各自 `reshape`（单消费者节点）：若共用一个视图，两个 gather 的梯度会先在视图里合并再回流，累加顺序一变，总梯度末位就变。
- **诊断标量批量回读**：7 次 `float()` → 1 次 `torch.stack([...]).tolist()`。
- **前向采样网格预计算**（`differentiable_defocus_uniformity.py`）：相机采样网格只依赖构造期定型的常量（相机坐标 / 配准平移 / 原点 / 逆 ramp / 子像素偏移 / pitch），与相位无关，故在 `__init__` 建一次，`predict()` 直接复用。

**行为中性（逐位）**：GPU 路径本身非确定（atomic 归约 + `grid_sampler` 并行求和，同代码两次运行校验和不同），故逐位对照一律走 **CPU** 路径（`torch.set_num_threads(6)`，两次运行逐位一致）：把改前源码树与工作树各跑一遍 model / anchored 各 12 步，**losses、最终 delta 校验和、末步诊断记录全部逐位相等**；`ShapeObjective.evaluate` 的三种模式（cv / shape / formation）在 **loss、7 个诊断标量、图像梯度** 上也全部逐位相等。

**CI 门禁**：`tests/ao_shaping/algorithm/test_multishape_objective_bit_exact.py` 把改前的布尔掩膜实现逐字复制为参考，钉住「逐位相等 + 原地改掩膜后仍逐位相等」。变异测试确认该门禁**能变红**：把共享 `flat` 视图这个已修的 bug 重新注入，`shape`/`formation` 两档立刻红（文件按 sha 复原）。

**代价**：两个文件行数未增（`differentiable_defocus_uniformity.py` 818 ≤ 基线 819），模块规模门禁**无需抬高基线**。

## 6. 结论与未解决

1. **光阑关闭态 side=1200 整形未复现**：72 个 mono12 块（每形状 24：cv 8 + continue 12 + dense 4）Jacobian 全部 `contradict`，末帧 CV 1.07–1.19 无收敛；重放 @0.2 ms CV 0.54–0.69、target_pearson ≤0.22，**无成形结构**。
2. **主导假设**：未压制零级（E0/E1≈1.2，`zero_radius=60` 只排除评价不物理消除）污染目标域；`locate()` 零级掩膜只修探测不修物理零级。
3. **曝光口径**：0.05 ms 近噪声下限（core_peak 127–150，伪影）；0.2 ms 可信（core_peak 364–420）但 CV 高。与 10-28 跨曝光/跨强度**不可直接对等**。
4. **效率**：GPU 快于 CPU（热机 2.3×，冷跑至 13×，时钟/预热主导）；瓶颈在 Python 算子调度（非 GPU 吞吐）；lr 0.01 已近离线收敛地板；anchored 前向复用为行为中性（loss 逐位一致），不构成稳定加速。**调度减负已落地（2026-11-10，§5.2.1/§5.4）：每步墙钟 GPU 12.58→7.88 ms（−37%）、CPU 44.49→26.46 ms（−40%），`cudaStreamSynchronize` 60→13 次/迭代、`aten::nonzero` 8→0，逐位行为中性**；原先「合并 roll+FFT / 粗网格 `fft256`」的方向**已证伪并撤回**（FFT+roll 仅 ~0.1 ms/迭代）。
5. **未解决项**：① 零级物理压制手段；② 模型↔硬件传递缺口（[diffopt_efficiency_20261108.md](../defocus_differentiable/diffopt_efficiency_20261108.md) 主约束——模型侧到 CV 0.012–0.043、硬件侧首块后无增益，方向/幅度不可迁移）；③ Jacobian `contradict` 根因（方向不可信，闭环无法改善）。

## 7. 下一步物理光路实验计划（结合 11-08 仿真结论）

11-08 `diffopt_efficiency_20261108.md` 关键结论（作为计划输入）：

- **主约束 = 模型↔硬件传递缺口**，不是每步毫秒或迭代数；效率优化收益在**模型侧吞吐**（更多起点 / 更大模式集 / 粗→细调度），**不缩小**传递缺口。
- **部署路径绑定 = `step_rms_cap=0.1`**（非 lr）：锚点不推进时总相位永久钉在 0.100 rad RMS、CV 停 0.580（lr 0.01/0.1/0.3 全同）；锚点每步推进时 lr=0.1 可到 0.012，但相位行程 ~1.0 rad RMS ≈ 10× cap。
- **裸 Adam lr=0.01 是病根**（300 步停 0.338），lr=0.1 到 0.080；但模块原生路径（cap 投影）掩盖 lr 收益。

**物理光路实验顺序（优先级从高到低）**：

1. **压制零级（前提，最高优先）**：光阑关闭态 E0/E1≈1.2 是当前主导退化。先物理恢复零级压制（重新对光阑 / 加 DC block，目标 E0/E1<0.35），再重做 side=1200 复现；若无法恢复，验证零级抑制算法（预测域扣除零级盘）能否让 CV 随零级扣除改善。**这是验证"未复现"是否由零级污染导致的关键一测。**
2. **提高 `step_rms_cap` + 调 lr**（结合 11-08）：cap 0.1 → 0.3（或逐周期增量上限），配 lr=0.1（11-08 结论）或 0.02（11-09 离线地板）。**前提是 Jacobian 方向可信**（当前全 `contradict`）——否则 cap/lr 调优无效，须先解决方向门禁。
3. **调度减负（两轮已落地，均行为中性）**：① anchored 前向复用把每步 4→3 次前向（§5.3）；② 2026-11-10 调度减负（§5.2.1/§5.4）：掩膜扁平索引 + 诊断标量批量回读 + 前向采样网格预计算 → GPU 每步 12.58→**7.88 ms**（−37%）、CPU 44.49→**26.46 ms**（−40%），`cudaStreamSynchronize` 60→13 次/迭代、`aten::nonzero` 8→0，**逐位行为中性**（CI 门禁 `test_multishape_objective_bit_exact.py`）。**原方向「合并 roll+FFT / 粗网格 `fft256`」已证伪并撤回**（FFT+roll 仅 ~0.1 ms/迭代）。剩余可查方向：元素级算子融合（改后仍 `aten::mul` 94 / `add` 70 / `div` 44 / `sum` 36 次/迭代）与采样点批量化（`cudnn_grid_sampler` 27 次/迭代）。
4. **曝光窗纪律**：replay 最终帧用 **0.2 ms**（core_peak 364–420，可信），不用 0.05 ms（近噪声下限伪影）；各形状最终帧指标以 0.2 ms 行为准。
5. **模型侧吞吐**（与物理光路解耦，可离线）：批量起点合成（8 起点 → 61.6 ms/步，每起点 7.7 ms，1.6×）+ 粗网格 `fft256`（−24%，仅粗阶段）+ 模式集 Noll 5–25（地板 0.009）。

**风险与门禁**：第 1 步是前提，未恢复零级前第 2 步（cap/lr）收益被传递缺口掩盖；第 2 步须过方向门禁（Jacobian 相关 ≥ 0.50，当前 ≤0.17）才有意义；第 4 步曝光口径必须固定，避免 0.05 ms 伪影污染结论。
