# 可微优化逐轮 SLM 相位回放与逐像素单调性诊断

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`scripts/slm/replay_diff_phase_monotonicity.py`](../../scripts/slm/replay_diff_phase_monotonicity.py)
> **数据/关联脚本**: [`scripts/slm/render_differentiable_opt_animation.py`](../../scripts/slm/render_differentiable_opt_animation.py)
> **数据/关联脚本**: [`scripts/slm/hw_differentiable_phase_opt.py`](../../scripts/slm/hw_differentiable_phase_opt.py)
> **数据/关联脚本**: [`src/ao_shaping/algorithm/signal_processing/differentiable_beam.py`](../../src/ao_shaping/algorithm/signal_processing/differentiable_beam.py)
> **运行环境**: 离线
> **说明**: 可微光束优化逐轮 SLM 相位的离线回放与逐像素单调性诊断：逐轮相位未落盘，但用 scripts/slm/render_differentiable_opt_animation.py 的重放设施逐位精确重建 121 帧相位栈（120 轮 loss 对 history.csv max_rel_diff=0、最终相位对 phase_final.npy max_abs_diff=0）。scripts/slm/replay_diff_phase_monotonicity.py 经 phase_model_to_panel 做中心对齐+最近邻尺寸缩放后，统计 active 像素（4.9%）的符号一致性：均值 0.611、单调(>=0.9)像素仅 0.06%；真实 lr 反事实重放显示当前 lr=0.05 近最优，0.08/0.12 使最终 loss 变差 8%/22%。结论：像素级相位更新不单调，不建议放大 lr，与 transfer fit 阶段 Zernike 放大 lr 处方不可混用。人工撰写。
<!-- provenance:end -->

> 类型：离线诊断报告 · 日期：2026-11-21 · 环境：离线（不操作任何硬件）
>
> 关联脚本：`scripts/slm/replay_diff_phase_monotonicity.py`（复用 `scripts/slm/render_differentiable_opt_animation.py` 的重放设施）
>
> 数据源：`data/debug/differentiable_opt/20261116_114424/`（可微光束优化硬件存档）
>
> 产物：`data/debug/differentiable_opt/phase_monotonicity_20261116.{json,png,npz}`

---

## 摘要

本报告回答两个问题：（1）**可微光束优化中，SLM 相位在经中心对齐 + 尺寸缩放到面板后，逐轮如何变化，能否回放？**（2）**逐像素的相位变化是否单调一致，据此能否调整学习率（lr）？**

核心结论：

1. **逐轮相位可精确回放**。存档未落盘逐轮相位（只有 `phase_initial/final` 两帧 + 逐轮标量历史），但用现有重放设施离线重建的 120 轮 loss 轨迹与 `history.csv` **逐位一致**（`max_rel_loss_diff=0`），最终相位与 `phase_final.npy` / `phase_model.npy` **逐位一致**（`max_abs_diff=0`）。因此逐轮相位（121 帧 × 1024²）是被精确重建的，不是近似或插值。
2. **逐像素变化并不单调**。在"真的在动"的 active 像素（占 4.9%）上，相邻增量的符号一致性均值仅 **0.611**，达到完全单调（≥0.9）的像素仅 **0.06%**——与 fit 阶段 Zernike 系数（后段余弦 0.998）形成鲜明对比。
3. **不应放大 lr，当前 lr=0.05 已是近最优**。真实反事实重放（同一冷启动）：lr=0.08 / 0.12 的最终 loss 比 lr=0.05 差 8% / 22%，且 120 轮内未达到存档目标；只有 lr=0.05 达到（第 98 轮）。诊断给出"方向不一致，不建议放大 lr，宜减小或保持"。

> **两阶段动力学本质不同**：transfer fit 阶段是低维（7 维 Zernike）、平滑弹道、方向高度一致 → 有放大 lr 的空间；可微整形阶段是像素级高维（~10⁵ 活跃像素）、更新方向不一致 → 放大步长只会放大震荡。两个阶段的 lr 处方**不可互相套用**。

---

## 1. 任务背景与范围

- **阶段**：可微光束整形（`DifferentiableBeamOptimizer`，Adam），区别于 transfer 管线的 `DifferentiableUniformityOptimizer` / `phase_lr=0.30` 和 fit 阶段的 `residual lr=0.03`。
- **存档运行**：`20261116_114424`，`iters=120`、`lr=0.05`、`lr_schedule=None`（flat）、`seed=0`、`m_grid=1024`、`n_pupil=256`、`pupil_center=[960,600]`、`pupil_radius=480`。
- **范围**：纯离线回放与诊断；不连接 SLM / CCD / 激光，不修改优化器默认参数。

## 2. 方法与口径

### 2.1 逐轮相位回放与校验

复用 `render_differentiable_opt_animation.py` 的 `load_run / load_main_module / build_model_inputs`，以存档的 `phase_initial.npy` 冷启动重建 `DifferentiableBeamOptimizer`，逐轮收集**每次迭代开始时**的 `current_phase`，得到 121 帧相位栈（含初始相位）。双重校验：

- loss 轨迹对照 `history.csv`：120 轮，逐位一致（`max_rel_loss_diff=0.0`）；
- 最终相位对照 `phase_final.npy` 与 `phase_model.npy`：`max_abs_diff=0.0`。

### 2.2 中心对齐 + 尺寸缩放

硬件实际下发的相位经 `hw_differentiable_phase_opt.py::phase_model_to_panel` 从模型网格（1024²）映射到 SLM 面板：以 `spec.pupil_center` 为锚点、`dp = 2·pupil_radius/n_pupil` 的最近邻缩放，瞳面块外置零。**可视化严格使用映射后的面板相位**（瞳面窗口裁剪）。

单调性统计在模型网格上计算：因该映射在瞳面块内是最近邻正比例拷贝（每像素增量符号与幅度不变），块外恒零已被 active mask 排除，故与在面板上统计**数学等价**；避免物化 1920×1208 的全零面板栈（见脚本 docstring）。

### 2.3 逐像素单调性定义

对相位栈取相邻增量 `d[k] = phase[k+1] − phase[k]`（119 步）。每像素的**符号一致性** = `max(正向步数, 负向步数) / 总步数`（0.5 = 随机翻转，1.0 = 完全单调；零增量按 0.5 中性计入）。仅统计 active 像素（`|phase_final − phase_initial| > 1e-3`），避免"没在动"的像素稀释。

- `frac_pixels_monotonic`：符号一致性 ≥ 0.9 的 active 像素占比；
- `mean_sign_consistency`：active 像素一致性均值；
- `per_iteration_direction_consistency`：每步在 active 像素上"增量符号与上一步相同"的占比（长 119）。

### 2.4 lr 反事实（真实重放，非代理）

以相同 `init_phase` / `seed` / `target` / `amplitude` 冷启动，对 lr ∈ {0.02, 0.05, 0.08, 0.12} 各重放 120 轮，记录最终 loss 与"达到存档最终 loss 水平（0.00013330）所需轮数"。这是**逐位可复现的真实重跑**，比上一轮 Zernike 诊断的一阶线性代理更强。

## 3. 关键数据

### 3.1 回放校验与单调性

| 指标 | 实测 | 含义 |
|------|------|------|
| loss 对照 history.csv | 120 轮 max_rel_diff = **0.0** | 逐位复现 |
| 最终相位对照存档 | max_abs_diff = **0.0** | 逐位复现 |
| `frac_active_pixels` | **4.90%** | 真正发生变化的像素占比 |
| `mean_sign_consistency`（active） | **0.6113** | 逐像素方向一致性低 |
| `frac_pixels_monotonic`（≥0.9） | **0.06%** | 几乎无像素单调推进 |
| `mean_total_change_active` | 0.2989 rad | active 像素总变化幅度 |
| 每步方向一致性（前 5 / 后 5） | 0.66→0.87 / 0.79→0.78 | 全程约 0.80，稳定但不高 |

### 3.2 lr 反事实扫描（真实重放）

| lr | 最终 loss | 达到目标(0.00013330)轮数 | 相对当前 |
|----|-----------|--------------------------|----------|
| 0.02 | 0.00013656 | 120 轮内未达到 | 更慢 |
| **0.05（当前）** | **0.00013330** | **98** | 基准 |
| 0.08 | 0.00014355 | 未达到 | 差 +7.7% |
| 0.12 | 0.00016209 | 未达到 | 差 +21.6% |

### 3.3 可视化（见产物 PNG）

`phase_monotonicity_20261116.png` 三子图：① 初始相位瞳面窗口（大尺度平滑结构）；② 迭代 0/11/…/120 共 12 帧面板相位快照（微结构逐步形成）；③ 符号一致性热力图——active 像素集中在瞳面孔内，颜色以紫蓝为主（一致性 0.5–0.7），直观印证"逐像素变化不单调"。

## 4. 结论与建议

**lr 建议（脚本输出）**：「逐像素方向不一致/已到噪声地板，不建议放大 lr，宜减小或保持。」

具体处方：

1. **不放大 lr**：当前 0.05 已达近最优，放大到 0.08/0.12 使最终 loss 变差 8%/22%；逐像素方向一致性仅 0.61，放大只会放大震荡。
2. **若要尝试减小**：0.02 在 120 轮内未达到存档目标（更慢但未必更差），可小步长 A/B 验证是否以时间换稳定；但不作为默认改动。
3. **两阶段分开管**：transfer fit 阶段的 Zernike 放大 lr 处方不适用于此像素级相位优化；两者不要混用同一 lr 策略。
4. **并行化另议**：GPU 利用率低（实测 15–19%）源于 kernel launch 开销而非算力；batch=2 吞吐近乎翻倍、batch≥4 退化。该问题属于吞吐优化，与本次 lr 收敛结论独立。

## 5. 口径边界（重要）

- **回放是精确重建，但非原始落盘**：逐轮相位由离线重放生成，校验 loss 与最终相位逐位一致；不能据此声称"硬件每轮实际下发的相位序列被记录"——中间帧是模型侧重放结果。
- **可微优化是仿真步**：存档 `history.csv` 的 loss 来自模型自身远场，不含测量锚定；本诊断不涉及真实 CCD 闭环的逐帧对应。
- **单一存档运行**：结论基于一次 run（lr=0.05 flat）；不同目标形状 / 光照 / 收敛阶段需重跑确认，不外推为通用最优 lr。
- **active 阈值敏感**：单调性统计依赖 `min_total_change=1e-3`；阈值改变会微调占比，但"方向不一致"的结论在宽范围内稳健（一致性 0.61 远离 0.9）。
- 全程离线，未操作任何硬件；未修改优化器默认学习率。

## 6. 任务项清单

- [x] T1 发现逐轮相位落盘/重放能力与中心对齐+尺寸缩放实现
- [x] T2 验证离线重放逐位复现存档（loss/相位 max_diff=0）
- [x] T3 实现逐轮相位回放脚本与面板映射 + 可视化
- [x] T4 实现逐像素单调性分析与 lr 三分支建议
- [x] T5 真实 lr 反事实扫描（4 个 lr，逐位可复现）
- [x] T6 在真实存档 run 上跑出结论、生成 JSON/PNG/NPZ 产物
- [x] T7 报告归档与溯源登记、ruff + pytest 校验

## 7. 校验记录

| 项目 | 命令 | 结果 |
|------|------|------|
| 单元测试 | `.venv\Scripts\python.exe -m pytest tests/ao_shaping/scripts/slm/test_replay_diff_phase_monotonicity.py --import-mode=importlib -q` | 11 passed |
| 代码风格 | `.venv\Scripts\python.exe -m ruff check scripts/slm/replay_diff_phase_monotonicity.py tests/ao_shaping/scripts/slm/test_replay_diff_phase_monotonicity.py` | All checks passed |
| 溯源同步 | `.venv\Scripts\python.exe scripts/sync_report_provenance.py --check` | clean |
| 端到端实跑 | `replay_diff_phase_monotonicity.py --run-dir data/debug/differentiable_opt/20261116_114424 ...` | 退出码 0，loss/phase max_diff=0，JSON/PNG/NPZ 产出 |
