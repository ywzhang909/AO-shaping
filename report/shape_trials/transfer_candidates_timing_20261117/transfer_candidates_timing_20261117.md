# Transfer 优化器 K 候选并行：离线 GPU 微基准与上机口径（2026-11-17）

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`scripts/slm/transfer_candidates_gpu_bench_20261117.py`](../../../scripts/slm/transfer_candidates_gpu_bench_20261117.py)
> **运行环境**: 离线（仅 CUDA，不占 SLM/CCD/激光；2026-11-17 生成）
> **说明**: Transfer 优化器 --candidates K 批前向的离线 GPU 微基准：真实 FrozenDefocusForward + DifferentiableUniformityOptimizer.update()，grid 32/64/128 × 4 形状 × K=1/2/4/8，单步 7–51 ms、K=8 相对 K=1 为 2.45–4.53×（次线性）；含端到端 301 s I/O 主导口径对比与 K=1 逐位一致证据。与 candidate_parallelism_20261117（另一模块）区分。人工撰写。
<!-- provenance:end -->

**结论先行**：transfer（beam quality GUI）的 `--candidates K` 批前向已落地，`K=1` 逐位一致。
纯优化器算子上，`update()` 单步在 RTX 5080 Laptop 上为 **7–51 ms**（grid 32→128、K 1→8），
K=8 相对 K=1 约 **2.5–4.5×**——次线性，非 K×。这个绝对量级比端到端硬件闭环的 301 s
小 4 个数量级（优化器本体在端到端里 <0.5%），所以"上机测端到端 K 加速"会被 SLM 显示等待 +
CCD 抓帧 + 模型拟合的 I/O 抖动淹没；**本轮以 GPU 微基准作为 K 代价的可靠证据**，端到端墙钟
只做"能否跑通"的验证性记录。

- 数据：`data/debug/transfer_candidates_gpu_bench_20261117/results.json`（48 配置点）
- 生成脚本：`scripts/slm/transfer_candidates_gpu_bench_20261117.py`（离线，仅 CUDA，**不占 SLM/CCD/激光**）
- K=1 逐位一致的独立证据：[`artifacts/k1_bitident_check.py`](artifacts/k1_bitident_check.py)（解析 5 步 + 锚定 5 步 + 物理 32 栅格 3 步，全 `np.array_equal`）
- 与 `report/shape_trials/candidate_parallelism_20261117.md` 区分：那份测的是
  `train_beam_shaping(n_candidates=)`（另一个模块）；本报告测的是 transfer 的 `--candidates K`。

## 1. 改了什么（不涉 MIL）

| 项 | 内容 |
|---|---|
| API | `DifferentiableUniformityOptimizer(..., n_candidates=K)`；CLI `--candidates K` |
| 语义 | 一次批前向 `(K,H,H)→(K,H,W)` + 逐候选损失求和 + **一次**反传 + Adam + 逐候选投影/去倾斜 |
| 对外 | 仍单相位：`current_phase`/`delta_phase`/`last_candidate_prediction` 取 argmin 的 best 候选（2D） |
| 逐位一致 | `K=1` 与缺省路径 loss/phase/delta/accepted/prediction 全等（golden 测试 + 独立脚本双证） |
| 关键修正 | `_project_increment`/`_remove_piston_and_tilt` 按候选归约到末两维 `keepdim=True`，避免 RMS cap 混成全局值 |
| 原生批 | `FrozenDefocusForward.predict` 直接支持 `(K,H,H)`，无需 `torch.vmap` |

涉及文件：`differentiable_candidates.py`、`differentiable_defocus_uniformity.py`、
`tools/slm/transfer/workflow.py`、`tools/slm/transfer/validate.py`、`test_differentiable_candidates.py`。
**未改**任何 MIL 文件（`model_in_loop_shaping.py` / `slm_model_in_loop.py` /
`shaping_controls.py` / `test_model_in_loop_shaping.py`）。

## 2. 微基准口径

- 计时对象：`DifferentiableUniformityOptimizer.update()`（model 模式），含真实
  `FrozenDefocusForward` 的 torch FFT 前向 + 损失 + 反传 + Adam + 投影——最贴近真实单步口径。
- CUDA 事件外再用 `torch.cuda.synchronize()` 计墙钟；每配置 2 次 warmup + 8 次计时，取中位数。
- 4 种目标形状（square/star/triangle/ring，相机采样窗口 13×13）× 3 网格（grid 32/64/128，
  fft 64/128/256）× K=1/2/4/8。峰值显存取 K=8。
- 小尺寸点存在 GPU kernel 启动抖动（个别 K=2 中位比 K=1 仅快几 ms），故按**量级/趋势**解读，
  不逐点比较。形状之间耗时接近（窗口固定 13×13，损失项计算量不随形状显著变化）。

## 3. 结果

按 4 形状取中位，每网格×K：

| grid / fft | K=1 | K=2 | K=4 | K=8 | 相对 K=1（K=8） | 峰值显存 (K=8) |
|---|---|---|---|---|---|---|
| 32 / 64  | 0.0078 s | 0.0107 s (1.36×) | 0.0144 s (1.84×) | 0.0230 s (2.94×) | 2.94× | 33 MiB |
| 64 / 128 | 0.0100 s | 0.0120 s (1.20×) | 0.0152 s (1.52×) | 0.0246 s (2.45×) | 2.45× | 37 MiB |
| 128 / 256| 0.0113 s | 0.0117 s (1.04×) | 0.0330 s (2.92×) | 0.0511 s (4.53×) | 4.53× | 54 MiB |

观察：

1. K=8 相对 K=1 为 2.45–4.53×，**次线性**：批量摊薄固定开销、提高 GPU 占用，但远不是免费。
2. 网格越大，K 的边际代价越明显（128 栅格上 K=4→8 由 2.92× 跳到 4.53×）。
3. 绝对单步 7–51 ms；即使 300 步也只有 2–15 s 量级，仍远小于端到端 I/O。
4. 显存占用温和（K=8 ≤ 54 MiB），不构成约束。

## 4. 端到端墙钟：为何 GPU 微基准才是有效口径

既有闭环（`20261117_202916`，square side=1000，端到端 301 s）的阶段拆解：

| 阶段 | 耗时 | 占比 |
|---|---|---|
| capture | 85.9 s | 28.6% |
| CCD_model 拟合 | 80.5 s | 26.7% |
| CCD_anchored | 100.3 s | 33.3% |
| fit | 23.1 s | 7.7% |
| audit | ~10 s | 3.3% |
| **优化器本体** | **≈1 s（0.027–0.067 s/步）** | **<0.5%** |

在这样 I/O 主导的回路里，K=1→8 的优化器增量（<1 s）会被 SLM 显示等待（0–200 ms/步）与
CCD 抓帧抖动整体淹没。因此"端到端测 K 加速"在统计上不可分辨；K 的真实代价必须用 §2–§3
的纯 GPU 微基准来量化。上机闭环的价值在于**验证 K>1 在真实 SLM/CCD 上跑通**，而非测加速。

## 5. 结论与去向

- `--candidates K` 可用，K=1 逐位一致，K>1 次线性加速代价已量化；是否默认 >1 取决于能否容忍
  2.5–4.5× 优化器耗时换非凸目标上的候选多样性收益（参照 `candidate_parallelism_20261117.md`
  的 1–2% 损失改善口径，但该收益数字属另一模块，不可直接套用，需在 transfer 目标上另测）。
- 设备将下线，端到端 K>1 验证性闭环排入下次在线窗口；命令：
  `python -m ao_shaping.tools.slm.transfer run --out <dir> --execute --candidates K`。
- GPU 微基准复现：`python scripts/slm/transfer_candidates_gpu_bench_20261117.py --out <path>`。
