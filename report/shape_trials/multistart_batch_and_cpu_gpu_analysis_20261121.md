# 可微光束整形：B=2 多起点批量并行与 CPU→GPU 整合分析

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`scripts/slm/sweep_differentiable_multistart_batch.py`](../../scripts/slm/sweep_differentiable_multistart_batch.py)
> **数据/关联脚本**: [`scripts/_common/differentiable_problem.py`](../../scripts/_common/differentiable_problem.py)
> **数据/关联脚本**: [`src/ao_shaping/algorithm/signal_processing/differentiable_beam.py`](../../src/ao_shaping/algorithm/signal_processing/differentiable_beam.py)
> **数据/关联脚本**: [`src/ao_shaping/algorithm/_torch_support.py`](../../src/ao_shaping/algorithm/_torch_support.py)
> **数据/关联脚本**: [`scripts/slm/hw_differentiable_phase_opt.py`](../../scripts/slm/hw_differentiable_phase_opt.py)
> **数据/关联脚本**: [`src/ao_shaping/utils/wavefront/calibration_terms.py`](../../src/ao_shaping/utils/wavefront/calibration_terms.py)
> **运行环境**: 离线
> **说明**: 可微光束整形 B=2 多起点批量并行 + CPU→GPU 整合分析：scripts/slm/sweep_differentiable_multistart_batch.py 复用 differentiable_problem 重建入口，一次 batch FFT 前向+一次反传+单 Adam(B,M,M)，每条 trajectory 独立峰值归一化、独立 copy 初相；真实存档 20261116_114424 跑 120 轮，batch trajectory 0 对照单条串行 loss max_rel_diff=6.56e-7、最终相位 max_abs_diff=3.34e-6（容差内 PASS，非 bit-exact）；双初相 RMS 差 0.05 证明非重复计算。端到端吞吐 3.37x（batch 1486 vs 串行 441 fields/s，其中混入去掉 D2H 同步/记账的收益，纯 batching 约 1.9x，不宣称实验帧率提升）。CPU→GPU 分析：P4 200 步全 GPU，P6b 闭环 5 步的 scipy map_coordinates / numpy 可搬 torch（grid_sample 有现成先例），但秒级 SLM/CCD I/O 主导延迟；建议仅做 grid_sample 重采样 + 消除重复 D2H 同步。人工撰写。
<!-- provenance:end -->

> 类型：离线吞吐与架构诊断报告 · 日期：2026-11-21 · 环境：离线（不操作任何硬件）
>
> 关联脚本：`scripts/slm/sweep_differentiable_multistart_batch.py`
>
> 数据源：`data/debug/differentiable_opt/20261116_114424/`（可微光束优化硬件存档）
>
> 产物：`data/debug/differentiable_opt/multistart_batch_20261116.{json,png}`
>
> 边界：本报告是**吞吐优化研究**，未修改优化器源码、未改任何默认参数、未操作 SLM / CCD / 激光。

---

## 摘要

本报告回答两个问题：（1）**把可微光束优化的一次前向/反传/Adam 步扩成 B=2 多起点批量（一次 batch FFT），数值是否仍与单条串行一致，吞吐收益多大？**（2）**可微优化流程里还有哪些 CPU 计算，能否搬进 torch 用 GPU 算？**

核心结论：

1. **B=2 批量数值一致（容差内，非 bit-exact）**。真实存档 120 轮，batch trajectory 0 对照单条串行：loss 轨迹 `max_rel_diff=6.56e-7`，最终相位 `max_abs_diff=3.34e-6`，均在 1e-4 容差内。差异来自 batch cuFFT plan 与单条略不同的归约顺序，**不是 bit-exact，不应谎称逐位相同**。
2. **多起点确实非重复计算**。trajectory 1 用固定种子扰动初相（RMS 差 0.0500 rad），两条 loss 轨迹独立演化：前 15 轮扰动起点 loss 明显更高，后段都收敛到 ~1.35e-4，最终 loss 略高（1.3679e-4 vs 1.3330e-4）——正好体现"多起点探索"的意义。
3. **端到端吞吐 3.37x，但要拆口径**。实测 B=2 batch 1486 fields/s vs 串行 441 fields/s（3.37x）。但这个对比中 batch 侧用的是精简 `batch_step`（无 D2H 同步/记账），串行侧是完整 `opt.update()`（每步 4 次标量 D2H + 相位拷贝 + 记账），**混入了"去掉同步开销"的收益**。纯 batching 的净收益是此前临时基准测得的 1.90x（B=2，前向+反向口径一致）。两个数字不可混谈。
4. **CPU→GPU 整合空间有限，且收益被 I/O 淹没**。真正的每轮 CPU 计算集中在 P6b 硬件闭环（默认仅 5 步）；P4 的 200 步纯模型迭代几乎全在 GPU。设备 I/O（SLM 上屏 `time.sleep(0.5)` + CCD 曝光/读帧）才是延迟主因。真正值得做的只有两件：`grid_sample` 替 `resample_ccd_to_model`（可微 + 代码质量）、消除重复 D2H 同步（纯提速）。

---

## 1. 任务一：B=2 多起点批量并行

### 1.1 设计要点与正确性口径

复用唯一问题重建入口 `scripts/_common/differentiable_problem.py::load_problem_from_run_dir`（口径与 `hw_differentiable_phase_opt.py::run_analyze_only` 一致），不改优化器源码。损失与 Adam 逐项复刻 `DifferentiableBeamOptimizer`：

1. 目标峰值归一化：`t = target.astype(float32) / max(t)`，batch 内共享一张目标；
2. 前向：`field = amp * exp(i·phase)`，`far = fftshift(fft2(field, dim=(-2,-1)), dim=(-2,-1))`，`intensity = |far|²`；
   - ⚠️ `fftshift` 必须显式 `dim=(-2,-1)`：默认平移所有维度，batch 张量上会把两条 trajectory 互换；这是实测验证过的技术处理（子 agent 冒烟时踩过）。
3. **每条 trajectory 独立峰值归一化**：分母取 `amax(dim=(-2,-1), keepdim=True)`，绝不跨 batch 取一个 max（否则损失串味）；
4. `losses = MSE(normalized_b, t)` 得每条标量，`losses.sum().backward()` 一次反传（逐元素梯度与单条串行一致）；
5. Adam：`torch.optim.Adam([phase_(B,M,M)], lr)`，逐元素状态，单 optimizer 覆盖 batch 即等价；读完整 `update()` 确认仿真步无裁剪/调度，无需额外复刻。

**初相 copy 正确性**：`build_optimizer` 用 `np.asarray(init_phase, dtype=float32)` 不复制，Adam 会原地改写传入数组。多起点用 `np.array(..., dtype=float32, copy=True)` 每条独立拷贝，避免污染。

**复现门禁**：trajectory 0（存档初相）对照 `build_optimizer(...)` 单条串行跑同样 iters，逐轮 loss `max_rel_diff` + 最终相位 `max_abs_diff` 任一超过 1e-4 即 JSON 标 FAIL、exit 2 且不报吞吐。

### 1.2 真实数据结果（120 轮，lr=0.05，存档 20261116_114424）

| 项目 | 实测 | 判定 |
|------|------|------|
| loss 轨迹 max_rel_diff | 6.557e-7 | PASS（< 1e-4） |
| 最终相位 max_abs_diff | 3.338e-6 | PASS（< 1e-4） |
| bit_exact | false | 如实标注，cuFFT plan 差异 |
| 两条初相 RMS 差 | 0.0500 rad | distinct=true，非重复计算 |
| traj0 / traj1 最终 loss | 1.3330e-4 / 1.3679e-4 | 两条独立收敛，扰动起点略高 |
| 单条串行 | 0.5436 s / 441.5 fields/s | 完整 `opt.update()`（含同步/记账） |
| B=2 batch | 0.1615 s / 1486.4 fields/s | 精简 `batch_step` |
| 端到端 speedup | **3.367x** | 见 §1.3 口径说明 |

可视化 `multistart_batch_20261116.png`：两条 loss 轨迹独立演化（traj1 前 15 轮明显更高、后段收敛），直观证明多起点有效。

### 1.3 ⚠️ 吞吐口径说明（重要，避免夸大）

3.37x 这个数字**不等于"批量 FFT 本身加速 3.37 倍"**：

- 串行侧每步是完整 `DifferentiableBeamOptimizer.update()`：含 4 次 `float(loss.detach().cpu())` 标量 D2H 同步、`current_phase` 的 `.detach().cpu().numpy()` 相位拷贝、以及 loss/grad 记账；
- batch 侧是精简 `batch_step`：`zero_grad` + 批量前向 + 一次反传 + `step()`，**不含**这些同步/记账。

因此 3.37x = 纯 batching 收益（≈1.90x，见此前一致的临时基准）× 去掉同步/记账开销的收益。若要在实际优化器里落地 B=2，需要同步保留这些记账路径，届时端到端收益会回落到接近 1.9x 量级。**本报告不宣称 SLM 实验帧率提升**——实验系统闭环端到端主要卡在 CCD 采集等待（硬件瓶颈），并行化仿真对实验系统总时长帮助更小。

### 1.4 适用场景

B=2（多起点批量）适合**批量扫描实验**：多个 lr / 种子 / 目标形状并行，吞吐近翻倍且数值一致。B≥4 在此前临时基准中反而退化（FFT 规模变大 + 显存带宽），不建议作为"提利用率"的手段。

---

## 2. 任务二：可微优化流程的 CPU 计算与 GPU 化分析

### 2.1 流程分层

| 类别 | 环节（文件:行号） | 频率 / 规模 | 现状 |
|------|------|------|------|
| **纯 GPU（无需搬）** | `DifferentiableBeamOptimizer.update()` 前向/损失/反向/Adam（`differentiable_beam.py:311-384`、`_torch_support.py:50-80`） | P4 每步；`propagation.py` / `objective.py` 无 numpy/scipy | 全 torch，已最优 |
| **每轮 CPU（P6b，5 步）** | `resample_ccd_to_model`（`hw_differentiable_phase_opt.py:576-605`，scipy map_coordinates 双线性） | 每硬件步 1 次；16MB 坐标 + 8MB 输出 | 可 GPU 化 ★ |
| 每轮 CPU（P6b） | `phase_model_to_panel`（:537-573，numpy 最近邻）+ `phase_to_slm_grayscale`（`phase_display.py:54-91`） | 每步 1 次；18MB×2 + 4.6MB | 可 GPU 化（收益低） |
| 每轮 CPU（P6b） | `prepare_frame`（`beam_metrics.py:647-674`，numpy median/clip）+ `capture_frames`（:994-1010，stack/mean） | 每步；40MB 级 | 可 GPU 化（中） |
| **GPU↔CPU 同步点** | `current_phase` `.detach().cpu().numpy()`（`differentiable_beam.py:253-255`）；`update()` 4 次标量 D2H（:353,354,382,383）；`_to_tensor` 每步 H2D（:64-68） | P4 每步 2–3 次 4MB D2H + 4 次标量 | 可减少（纯提速） |
| **一次性预处理（不必搬）** | `build_model_spec` / `build_source_amplitude` / `build_target`（:408-534）、`build_initial_phase`（GS 50 次 FFT，:2098-2143） | 单次 O(10⁶) 元素 | 不在循环内，可忽略 |
| **设备 I/O（搬不了）** | SLM `display_data` + `time.sleep(0.5)`（`driver.py:240-244`）；CCD 曝光自适应最多 6 档×3 帧 + 每步最多约 23 次 `get_numpy_image`（:994-1010）；SDK 调用 | 每硬件步，秒级 | 必然主线程 CPU，**延迟主因** |

### 2.2 可 GPU 化候选与优先级

1. **★最高：`resample_ccd_to_model` → torch `grid_sample`**。项目已有两处现成先例：`calibration_terms.py:427-450`（`_resample`，bilinear + align_corners=True）与 `src/ml/sim2f/measurement.py:124-160`（`F.affine_grid` + `F.grid_sample`，最贴近"相机域→模型域"缩放）。注意口径：本脚本用 `mode="constant", cval=0`，对应 `padding_mode="zeros"`；需核对 `align_corners` 语义避免半像素偏移。收益不只是速度——重采样进入 torch 图后可微，且省掉 scipy↔torch 转换。
2. **中：`prepare_frame` / `capture_frames` 的 numpy 统计 → torch**。纯提速，无数值口径风险。
3. **低：`phase_model_to_panel` + 灰度转换 → torch 索引 / `grid_sample(mode="nearest")`**。输出最终必须 `.cpu().numpy()` 交 SDK，**只能省计算不能省同步**；且最近邻映射口径必须与现有实现逐位一致（见单调性报告 §2.2），改动需配逐位对照测试。
4. **纯提速：消除重复 D2H 同步**。P4 每步 `opt.current_phase.copy()` 被调用 2–3 次（连续取同一相位），复用一次副本即可；振幅驻留 GPU 免 `_to_tensor` 每步 H2D；4 次标量 D2H 可合并。

### 2.3 关键判断：CPU 计算不是主要瓶颈

- **P4 纯模型迭代**（200 步）：CPU 侧只有同步与记账；GPU 利用率低（实测 15–19%）源于 kernel launch 开销，不是 CPU 计算——杠杆在 batching（§1）或 `torch.compile` / CUDA Graph，不在"把 CPU 计算搬 GPU"。
- **P6b 硬件闭环**（5 步）：单步有 0.5 s SLM 上屏 sleep + 最多约 23 帧 CCD 曝光/读取，**秒级 I/O 淹没毫秒级 CPU 计算**。把 numpy/scipy 搬 GPU 省不下主要延迟；现有日志只对 torch 部分计时，未单独统计曝光自适应/帧读取/上屏等待，无法从日志给出精确 CPU vs I/O 占比（但从调用次数与数组规模看 I/O 主导）。
- 已知反例警示：`report/shape_trials/aperture_closed_20261109/...` 记载 grid_sampler 并行求和的 GPU 非确定性——逐位对照类检查仍应走 CPU 口径。

---

## 3. 结论与建议

1. **批量扫描可用**：B=2 多起点批量在 1e-4 容差内与串行一致（非 bit-exact），适合批量扫 lr / 种子 / 目标，吞吐近翻倍至 3.37x（口径见 §1.3，实际优化器落地后约 1.9x 量级）。
2. **CPU→GPU 优先级**：只做两件——`grid_sample` 替 `resample_ccd_to_model`（可微 + 代码质量）、消除重复 D2H 同步（纯提速）；`prepare_frame` 居中；`phase_model_to_panel` 收益低且需守住最近邻口径。
3. **不要指望 CPU→GPU 解决实验系统延迟**：实验系统端到端主要卡 CCD 采集与 SLM 上屏等待，并行化仿真对总时长帮助有限。
4. **与 lr 结论独立**：本报告是吞吐问题；此前"可微相位 lr=0.05 近最优、不放大"是收敛问题，两者不混谈。

## 4. 口径边界（重要）

- **非 bit-exact**：batch 与串行的 6.56e-7 / 3.34e-6 差异来自 cuFFT plan 归约顺序；容差内一致，但不宣称逐位相同。
- **吞吐口径混合**：3.37x 含"去掉同步/记账"收益，纯 batching 约 1.90x；不宣称实验帧率提升。
- **CPU vs I/O 占比未实测**：现有日志未单独计时硬件 I/O，I/O 主导是依据调用次数与数组规模的推断，非计时实证。
- **单一存档**：数值结论基于一次 run（lr=0.05 flat，1024² 网格）；其他目标/网格需重跑确认。
- **未改默认参数**：优化器源码、`build_optimizer`、lr 默认值均未动；全程离线，未操作任何硬件。

## 5. 任务项清单

- [x] T1 确认批量设计基底（共享问题入口、峰值归一化口径、Adam 批量状态）
- [x] T2 实现 B=2 多起点批量脚本 + 8 项合成测试
- [x] T3 真实存档 120 轮 repro 门禁 + 吞吐对比 + JSON/PNG 产物
- [x] T4 发现并核实 CPU 计算环节（explorer + orchestrator 逐行核验）
- [x] T5 输出 CPU→GPU 整合结论与优先级
- [x] T6 报告归档 + 溯源登记 + ruff/pytest 全量校验 + 提交

## 6. 校验记录

| 项目 | 命令 | 结果 |
|------|------|------|
| 单元测试 | `.venv\Scripts\python.exe -m pytest tests/ao_shaping/scripts/slm/test_sweep_differentiable_multistart_batch.py --import-mode=importlib -q` | 8 passed |
| 代码风格 | `.venv\Scripts\python.exe -m ruff check scripts/slm/sweep_differentiable_multistart_batch.py tests/ao_shaping/scripts/slm/test_sweep_differentiable_multistart_batch.py` | All checks passed |
| 端到端实跑 | `sweep_differentiable_multistart_batch.py --run-dir data/debug/differentiable_opt/20261116_114424 --iters 120 --batch 2 --device cuda ...` | 退出码 0，repro PASS（6.56e-7 / 3.34e-6），JSON+PNG 产出 |
| 溯源同步 | `.venv\Scripts\python.exe scripts/sync_report_provenance.py --check` | clean |
| 全量测试 | `.venv\Scripts\python.exe -m pytest tests/ao_shaping/scripts/slm/ tests/ao_shaping/tools/slm/transfer/test_debug_weights.py --import-mode=importlib -q` | 38 passed（30 既有 + 8 新增） |
