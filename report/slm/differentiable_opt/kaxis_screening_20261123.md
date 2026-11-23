# 冻结问题 `phys_20261116_01` 可微整形加速方案速度/质量公平筛选

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`scripts/slm/sweep_kaxis_frozen_20261123.py`](../../../scripts/slm/sweep_kaxis_frozen_20261123.py)
> **数据/关联脚本**: [`scripts/_common/differentiable_problem.py`](../../../scripts/_common/differentiable_problem.py)
> **数据/关联脚本**: [`src/ao_shaping/algorithm/signal_processing/differentiable_shaping.py`](../../../src/ao_shaping/algorithm/signal_processing/differentiable_shaping.py)
> **运行环境**: 仅 GPU 离线（RTX 5080 Laptop，1024² 网格 cuda）；未占用 SLM / CCD / 激光，全程无实验动作
> **说明**: 冻结问题 phys_20261116_01 上可微整形加速方案公平筛选（K 轴候选并行 + gate 复现 + 参照变体）：gate 复现逐位一致（max_rel_dev=0.0 ≤ 0.001, PASS）。K 轴 n_candidates∈{1,4,8}（Adam lr=0.05, 120 迭代）**不构成加速**：K>1 终态 cv 仅改善 ~0.05%（0.28381→0.28365，种子噪声 2–4 倍）、墙钟 +17%（K=4）~+60%（K=8），且 iters_to_k1_cv=115 三者相同（不提前收敛）⇒ 严格支配、K 轴淘汰；与 candidate_parallelism_20261117（grid 256 square 仅 0.07% 收益）一致。参照变体分开报告：L-BFGS（lr=1.0）第 5 迭代即达 K=1 终态 cv（≈0.67 s vs K=1 全程 5.91 s，快 ≈8.9×），终态 cv -0.15%、energy +2.5% ⇒ **算法轴加速候选**；cosine 调度质量 -0.42% 无收益。入围物理复验配对 = K=1 Adam 基线 vs L-BFGS 早停。人工撰写。
<!-- provenance:end -->

**日期**：2026-11-23（UTC 14:05 完成）
**设备**：`cuda`（RTX 5080 Laptop）；**未占用 SLM / CCD / 激光，全程无实验动作**，无采集，未修改原始数据。
**结论一句话**：在冻结问题口径上，候选并行（K>1）**不构成加速**（质量仅 ~0.05% 改善、墙钟 +17%~+60%，且不提前收敛），**淘汰**；算法轴的 **L-BFGS 才是加速杠杆**（5 迭代即达 K=1 终态质量，≈0.67 s 墙钟 vs 5.91 s），与 K 轴分开报告，作为**入围加速候选**待物理复验。

---

## 1. 目的与冻结问题

用冻结离线问题公平筛选 SLM 可微整形**加速方案**的速度与整形质量，并据此提出入围方案的物理复验最小设计。筛选**零设备、不执行采集、不修改原始数据**；仿真耗时与模型损失**不替代** CCD 验收。

冻结问题为 `phys_20261116_01`（`data/debug/differentiable_opt/phys_20261116_01/run_config.json`，经 `load_problem_from_run_dir` 离线重建，纯模型侧、不触发采集）：

| 参数 | 值 |
|---|---|
| 网格 | 1024² |
| 目标 | square，`target_side_px=60.0` |
| 初始相位 | `init_phase=gs`，`gs_iters=50` |
| 基线 | `lr=0.05` / `iters=120` |
| 传播 | `propagation="fft"` |
| 模型 | `model_pupil_samples=256`，`model_padding=4` |

质量口径与 `sweep_differentiable_lr.py::run_arm` 对齐：`mask=target>0.0`、`far_field_intensity`、`measure_spot`、`energy_in_target`、`cv_in_target`、`strehl_like`、`pred_fwhm_px`。**gate 与 K 轴损失不同，只比较物理质量与墙钟，不跨损失比较。**

## 2. 方法与公平性

筛选脚本：`scripts/slm/sweep_kaxis_frozen_20261123.py`（`--plot`）。变体冻结如下：

- **gate_repro**：`build_optimizer(lr=0.05, seed=0)` 跑 120 步，门 `max_rel_dev <= 0.001`（对照 `lr_sweep_20261116/lr_sweep.json` 基线）。
- **K 轴**：`train_beam_shaping`，固定 `target / amplitude / init_phase / grid_size / propagation="fft" / iterations=120 / lr=0.05 / seed`，仅变 `n_candidates ∈ {1,4,8}`（`k1_seed0` + `k4_seed0/1/2` + `k8_seed0/1/2`）；内部基线为 **K=1**。
- **参照变体**：`ref_lbfgs`（`optimizer=lbfgs`，`lr=1.0`，K 回退 1）、`ref_cosine`（`optimizer=adam`，`lr=0.1`，`lr_schedule="cosine"`）。**参照变体与 K 轴分开报告**。

K 轴比较包含候选扰动（K>1 与显式 GS 初相位叠加 `0.1*randn`）与最优候选选择，因此是「候选并行 + 择优」语义，不是单纯 batch 吞吐。

## 3. 结果

### 3.1 Gate 复现（PASS）

| 项 | 值 |
|---|---|
| `final_loss` | 1.3330099e-04 |
| `cv_in_target` | 0.2919478 |
| `energy_in_target` | 0.7887615 |
| `pred_fwhm_px` | 31.16333 |
| `strehl_like` | 0.0289914 |
| `wall_s` | 7.6964 |
| **`max_rel_dev`** | **0.000000**（各键均为 0.0） |
| `repro_tol` | 0.001 |
| **判定** | **PASS** |

gate 与 `lr_sweep_20261116/lr_sweep.json` 基线**逐位一致**（相对偏差 0.0），基线复现成立。

### 3.2 K 轴（候选并行）——淘汰

| 变体 | wall_s | per_iter_s | cv | energy | fwhm_px | strehl | iters_to_k1_cv |
|---|---|---|---|---|---|---|---|
| **k1_seed0（内部基线）** | 5.9141 | 0.04928 | **0.28381** | 0.83383 | 31.323 | 0.033668 | 115 |
| k4_seed0 | 7.1034 | 0.05920 | 0.28362 | 0.83401 | 31.332 | 0.033551 | 115 |
| k4_seed1 | 6.8502 | 0.05708 | 0.28367 | 0.83373 | 31.329 | 0.033595 | 115 |
| k4_seed2 | 6.8570 | 0.05714 | 0.28367 | 0.83363 | 31.329 | 0.033635 | 115 |
| k8_seed0 | 9.4611 | 0.07884 | 0.28362 | 0.83401 | 31.332 | 0.033551 | 115 |
| k8_seed1 | 9.4095 | 0.07841 | 0.28370 | 0.83403 | 31.331 | 0.033563 | 115 |
| k8_seed2 | 9.5262 | 0.07939 | 0.28363 | 0.83407 | 31.333 | 0.033568 | 115 |

相对 K=1 汇总：

| 对比 | cv 变化 | 墙钟变化 | 收敛迭代 |
|---|---|---|---|
| K=4（均值，种子 0–2） | −0.000155（**−0.05%**） | 6.94 s（**+17.3%**） | 115（同 K=1） |
| K=8（均值，种子 0–2） | −0.000158（**−0.05%**） | 9.47 s（**+60.1%**） | 115（同 K=1） |

**判读**：

- K>1 的终态 cv 仅改善 **~0.05%**（绝对 0.00015–0.00016），为种子噪声（K=4 种子散布 0.00005、K=8 种子散布 0.00008）的 2–4 倍——**边缘、非实质**。
- `energy / fwhm / strehl` 变化均在 ±0.1% 内，无任何物理质量轴实质改善。
- `iters_to_k1_cv = 115` **对 K=1/4/8 全部相同**：K>1 并不提前收敛，候选择优未找到更优解。
- 墙钟随 K 单调上涨：K=4 +17%，K=8 +60%。

**结论**：在冻结问题（square 单盆地目标）上，K>1 是**「用 ~17–60% 更多时间换取 ~0.05% 质量」的严格支配取舍**，不构成加速 ⇒ **K 轴淘汰**。此结论与 `candidate_parallelism_20261117.md`（grid 256 / 300 iter / seed 42：square 目标仅 0.07% 收益、8 候选慢 4.6–5.8×）**一致**——冻结问题为 square 单盆地，K>1 无质量收益是预期行为；本问题因 1024² 网格每候选并行工作更多，慢放比收窄到 1.6×（而非 4.6–5.8×），但质量收益同样归零。

### 3.3 参照变体（算法轴，分开报告）——L-BFGS 是加速杠杆

| 变体 | wall_s | per_iter_s | cv | energy | fwhm_px | strehl | iters_to_k1_cv |
|---|---|---|---|---|---|---|---|
| **ref_lbfgs** | 15.9749 | 0.13312 | **0.28337** | **0.85484** | 31.049 | 0.033072 | **5** |
| ref_cosine | 5.4087 | 0.04507 | 0.28500 | 0.83099 | 31.369 | 0.034959 | null |

**判读**：

- **ref_lbfgs**：`iters_to_k1_cv = 5` ⇒ **L-BFGS 在第 5 迭代即达 K=1 的终态 cv（0.28381），仅 ≈0.67 s 墙钟（5×0.133 s）**，比 K=1 全程（5.91 s）**快 ≈8.9×**。且其终态（120 迭代）质量更优：`cv=0.28337`（−0.15%）、`energy=0.85484`（**+2.5%**，能量更集中于目标）、`fwhm=31.05`（更窄）。代价是每迭代更贵（0.133 s vs 0.049 s），但**迭代数大幅下降**是主导。
- **ref_cosine**：`cv=0.28500` 比 K=1 差 0.42%，`energy=0.83099` 差 0.34%，且 `iters_to_k1_cv=null`（全程未达 K=1 终态 cv）。墙钟 5.41 s（0.91×）略快，但**质量无收益** ⇒ 不作为加速候选。

**结论（算法轴）**：**L-BFGS 是真正的加速候选**——以算法变化（Adam→L-BFGS）换取「少迭代达质量」，5 迭代即达基线质量且终态更优。这是**算法变化**，按公平性纪律与 K 轴**分开报告**，需独立验证，不能与 K 轴结论混合。

## 4. 入围 / 淘汰

| 方案 | 判定 | 依据 |
|---|---|---|
| K=1 Adam（lr=0.05，120 迭代） | **入围（基线）** | 冻结问题最稳妥；质量/墙钟基准 |
| K=4 / K=8 候选并行 | **淘汰** | +17%~+60% 墙钟换 ~0.05% cv，严格支配 |
| L-BFGS（lr=1.0，早停 5 迭代） | **入围（加速候选，算法轴）** | 5 迭代达基线质量（≈0.67 s vs 5.91 s），终态 energy +2.5% |
| Cosine 调度（Adam lr=0.1） | 不入选 | 质量 −0.42%，无加速收益 |

**加速方案结论**：候选并行（K>1）**不加速**；**算法轴（L-BFGS）才是加速方向**。入围加速候选 = **L-BFGS 提前停止**，与 **K=1 Adam 基线**构成物理复验配对。

## 5. 物理复验最小设计（配对比较，待窗口授权）

依据纪律：**配对比较速度与整形质量，控制初始相位、曝光、位深、目标与停止条件；按配对顺序比较并从不同起点重复**。方案：

- **方案 A（基线）**：K=1 Adam，`lr=0.05`，120 迭代（离线墙钟 5.91 s，cv 0.2838）。
- **方案 B（加速候选）**：L-BFGS，`lr=1.0`，**停止条件 = 达 A 终态 cv**（离线为迭代 5，≈0.67 s）；等价可写 `--iters 5` 或按 cv 早停。
- **控制变量**：初始相位 GS（`gs_iters=50`）、曝光/位深（Mono12、同曝光）、目标掩膜（square 60 px）、停止条件（达 A 终态 cv）。
- **配对与重复**：A→B 与 B→A 各一组，从**不同起点**（不同 seed / 初始相位扰动）重复 **≥2** 次（共 ≥4 次运行）；按配对顺序比较。
- **测量**：CCD 远场整形质量（`cv_in_target`、`energy_in_target`、`strehl_like`、`pred_fwhm_px`）+ **墙钟**（含上屏与采集，不含离线编译）。
- **判决**：B 达 A 终态质量的墙钟**显著短**（配对时间比 < 1/3，即 <~2 s 量级）**且** B 终态质量不劣于 A（cv 相对差在容差内）⇒ **B 入围**；否则 B 淘汰、维持 A。

> 物理复验需一次实验窗口（一次 SLM open/close、Mono12、`--wait-until-idle 600`、段间冷却 ≥200 s、每段独立 `--out`、采集期间不判读）；**未获授权不执行**。

## 6. 产物与复现

- 调试产物：`data/debug/differentiable_opt/kaxis_screening_20261123/screening.{json,csv}`。
- 报告产物：`report/slm/differentiable_opt/artifacts/kaxis_screening_20261123/kaxis_screening.json`、`kaxis_cv_vs_iter_20261123.png`。
- 复现：`python scripts/slm/sweep_kaxis_frozen_20261123.py --plot`（gate PASS、max_rel_dev=0.0）。
