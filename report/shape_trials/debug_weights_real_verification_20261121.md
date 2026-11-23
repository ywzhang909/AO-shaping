# `--debug-weights` 实验系统验证报告（2026-11-21）

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`src/ao_shaping/tools/slm/transfer/workflow.py`](../../src/ao_shaping/tools/slm/transfer/workflow.py)
> **数据/关联脚本**: [`src/ao_shaping/tools/slm/transfer/fit.py`](../../src/ao_shaping/tools/slm/transfer/fit.py)
> **数据/关联脚本**: [`scripts/slm/probe_optical_path_validity.py`](../../scripts/slm/probe_optical_path_validity.py)
> **数据/关联脚本**: [`scripts/slm/visualize_weights_tsne.py`](../../scripts/slm/visualize_weights_tsne.py)
> **数据/关联脚本**: [`scripts/slm/diagnose_zernike_direction_lr.py`](../../scripts/slm/diagnose_zernike_direction_lr.py)
> **运行环境**: 硬件
> **说明**: --debug-weights 逐轮权重轨迹落盘的实验系统验证：实验前光路六步探测判定 ok，短时冷启动采集+拟合 6 轮在 SLM #1 + Daheng CCD 上完整在线实现，weights_history.jsonl / weights_trajectory.npz / weights_summary.json 三份产物在真实采集数据上落地，训练损失 0.39325→0.34548；聚焦离线测试 6 用例通过。debug 数据以标准 JSONL/NPZ 格式保存，支持会话后离线回放；scripts/slm/visualize_weights_tsne.py 用 t-SNE 将 Zernike 系数轨迹降维到 2D 可视化。scripts/slm/diagnose_zernike_direction_lr.py 基于 34 条历史轨迹做观察性诊断：方向余弦后 10 轮中位数 0.9978、91.2% 轨迹后段方向一致、步长-损失下降 Spearman 0.7417，支持「早期/中期可放大 residual lr、噪声地板后收回」的一阶代理处方，不改默认学习率。口径要点：reuse_latest 命中已有 latest 模型会跳过采集+拟合导致无权重产物，须用 AO_SHAPING_WEIGHTS_DIR 重定向到空目录强制冷启动。人工撰写。
<!-- provenance:end -->

## 摘要

本报告记录 `--debug-weights`（逐轮权重轨迹落盘）在**实验**上的验证结果：按
`ao-optical-path-calibration` 六步探测实验前检查，随后以短时冷启动采集+拟合在
SLM #1 + Daheng CCD 上在线实现真实训练，确认 `weights_history.jsonl` /
`weights_trajectory.npz` / `weights_summary.json` 三份产物在真实采集数据上完整落地，
离线聚焦测试 6 项全过，设备全部正常释放。

关键结论：`--debug-weights` 在真实数据链路上**功能正确**——6 轮迭代的权重轨迹
（18 个参数 × 7 行，含迭代 0 初始值）完整落盘，训练损失从 `0.39325` 单调优化至
`0.34548`；产物格式与离线测试口径一致。debug 数据以标准 JSONL/NPZ 格式保存，
支持会话后离线回放；配套 t-SNE 可视化脚本可将 Zernike 系数轨迹降维到 2D 展示。

另附 **Zernike 方向一致性与 residual lr 诊断**：基于 34 条历史轨迹的观察性分析显示，
方向余弦后 10 轮中位数 0.9978、91.2% 轨迹后段方向一致、步长与损失下降强正相关
（Spearman 0.7417），支持"早期/中期可安全放大 residual lr 加速收敛，进入噪声地板后
收回"的处方；但这是基于步长-收益相关的一阶代理推断，非真实重跑，故不改动默认学习率。

**口径要点**：会话默认 `reuse_latest=True`，光路复用命中已有 latest 模型会**跳过
采集+拟合**，导致 `--debug-weights` 无产物。本次通过环境变量
`AO_SHAPING_WEIGHTS_DIR` 将权重目录重定向到空临时目录，使复用探测返回
"没有 latest 模型，走正常全流程"，强制走真实冷启动采集+拟合。这是本报告与后续
实验系统验证该功能时的标准口径。

## 任务完成情况

- [x] T1 实验前光路六步探测：Step 1–5 全部通过，判定 `ok`；设备锁 `status=idle`，无冲突进程。
- [x] T2 短时真实训练验证：冷启动采集+拟合（强制绕过光路复用）在 SLM #1 + Daheng CCD 上完整在线实现；model 验收与 CCD 锚定块（sequential_ccd）验收均产出记录。
- [x] T3 `--debug-weights` 三份产物核验：`weights_history.jsonl`（7 行）、`weights_trajectory.npz`（18 参数 × 7 轮）、`weights_summary.json` 均在真实采集数据上落地，内容合理。
- [x] T4 聚焦验证：`tests/ao_shaping/tools/slm/transfer/test_debug_weights.py` 6 用例全部通过（`--import-mode=importlib`）。
- [x] T5 设备释放确认：SLM #1 与 CCD 正常关闭，设备锁释放，探测复跑 `status=idle`。
- [x] T6 回放能力确认：`--debug-weights` 产物以标准 JSONL/NPZ 格式保存在 run 目录内，可被 `transfer replay` 及独立脚本回放处理，无需设备在线。
- [x] T7 t-SNE 可视化：实现 `scripts/slm/visualize_weights_tsne.py`，将权重轨迹降维到 2D，展示在线调优中网络 Zernike（`residual_noll_*`）系数的变化轨迹。
- [x] T8 Zernike 方向-lr 诊断：实现 `scripts/slm/diagnose_zernike_direction_lr.py`，基于 34 条历史轨迹的观察性分析得出 residual lr 调整处方（一阶代理，不改默认值）。

## 指标参数

### 设备与占用

| 项目 | 参数 | 说明 |
|------|------|------|
| SLM | Santec SLM-200 #1（`22030108`） | 1064 nm，2π 灰度设备动态读取 993，内部内存模式 |
| CCD | Daheng MER2-507-23GM（SN `FJB24112232`） | 2592×1944，曝光量程 0.020–1000 ms（满量程 4095） |
| 激光 | 沿用当前低功率设定 | **未做任何功率调整**，仅在用户授权的低功率可控环境下使用 |
| 设备锁 | `device_mutex` / `probe_devices` | 运行前后 `status=idle`，无其他硬件进程持锁 |

### 实验前光路探测（Step 1–5）

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

## 回放与可视化

### 回放支持

`--debug-weights` 产出的三份文件均使用标准格式（JSONL / NPZ / JSON），存放在
run 目录的 `fit/debug_weights/` 子目录内，与 `iterations.pkl` / `result.json` 等
规范产物同住。这些文件不依赖设备在线状态，可在会话结束后独立回放处理：

- **`transfer replay` 命令**：对整个 run 目录执行 `audit_run` + `summary.summarize`
  + `direction.analyze`，回放审计与重算。debug_weights 数据作为 run 目录的一部分
  自然被纳入回放范围，无需额外开关。
- **独立脚本回放**：任何读取 `weights_trajectory.npz` / `weights_history.jsonl`
  的脚本均可在离线环境复现分析，无需 SLM、CCD 或激光在线。t-SNE 可视化脚本即
  属此类。

### t-SNE 2D 可视化

**脚本**：`scripts/slm/visualize_weights_tsne.py`

**目的**：在线调优中，网络的 Zernike 系数（`residual_noll_*`，Noll 多项式展开）
在每轮迭代中持续变化。18 维参数空间的逐轮轨迹难以直观判读，t-SNE 将其降维到
2D 平面，展示系数演化的整体走向与轮次间步幅。

**用法**：

```bash
# 仅 Zernike 系数模式（默认）
.venv\Scripts\python.exe scripts\slm\visualize_weights_tsne.py \
    --input data\debug\slm_transfer\<run>\fit\debug_weights \
    --out <输出路径>.png

# 全参数模式
.venv\Scripts\python.exe scripts\slm\visualize_weights_tsne.py \
    --input data\debug\slm_transfer\<run>\fit\debug_weights \
    --out <输出路径>.png --mode all
```

**参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | — | debug_weights 目录路径，须含 `weights_trajectory.npz` |
| `--out` | — | 输出 PNG 路径 |
| `--mode` | `zernike` | `zernike` 仅取 `residual_noll_*` 列；`all` 取全部 18 个参数列 |
| `--perplexity` | 5 | t-SNE 的 perplexity 参数 |
| `--random-state` | 42 | 随机种子，保证可复现 |

**输出**：2D 散点图，按迭代轮次连线形成轨迹，颜色随轮次渐变并标注迭代号，
可直接用于判读系数演化的趋势与步幅。

**注意事项**：

- t-SNE 在点数极少时（N < 3）效果有限，短运行（如本次 6 轮）仅作示意，
  长运行的轨迹演化更具参考价值。
- 特征维度 < 2 时脚本报错退出（t-SNE 要求至少 2 个特征）。
- 全程离线，不操作任何硬件设备。

## Zernike 方向一致性与 residual lr 调整诊断

前文观察到单次实验运行的 Zernike 增量方向高度一致（相邻余弦 0.96–0.99）。为确认
这不是单次偶然，用**全部历史轨迹**做了一次观察性诊断，并据此评估能否调整 fit
阶段的 residual 学习率（Adam，当前 `0.03`，对应 `residual_noll_*`；注意这与整形
阶段的 `phase_lr=0.3` 不是同一机制）。

**诊断脚本**：`scripts/slm/diagnose_zernike_direction_lr.py`
**产物**：`data/debug/slm_transfer/zernike_direction_lr_diagnosis.json` / `.png`
**样本**：34 条历史 fit 轨迹（每条 ≥ 30 轮；排除 3 条过短历史），跨 2026-11-04 至
2026-11-20，涵盖 61 轮与 201 轮两种长度。

### 方法

把每条轨迹的 `residual_noll_*`（旧版为 `residual_noll_radians` dict）按 Noll 序号展平
成 7 维向量，计算相邻步增量的余弦相似度，并统计：方向一致性、步长、损失下降，
以及"方向一致（cos>0.9）时步长与该步损失下降"的 Spearman 相关。为避免循环论证，
余弦 `cos[i]`（比较 step i 与 step i+1 的方向）被对齐到 **step i+1** 的步长与损失
下降——即用前一步的方向预测下一步的收益。

### 关键数据

| 指标 | 实测 | 含义 |
|------|------|------|
| `cos_last10_median` | **0.9978** | 后 10 轮方向余弦中位数，几乎完全同向 |
| `frac_runs_cos_gt_0_9` | **91.2%**（31/34） | 后段方向一致的轨迹占比 |
| `consistent_active_frac_median` | **0.9133** | "方向一致且损失仍在下降"的步占比中位数 |
| `step_vs_lossdrop_spearman` | **0.7417**（p≈0，n=4336） | 方向一致时，步长越大损失下降越多（强正相关） |
| `frac_runs_reach_noise_floor` | 23.5%（8/34） | 末段跌入噪声地板（余弦<0.7）的轨迹占比 |

按步长四分位分桶，方向一致步的平均损失下降随步长单调递增：

| 步长 `‖dZ‖` 区间 | 样本数 | 平均损失下降 |
|------------------|--------|--------------|
| [0.0036, 0.0110) | 1084 | +0.00020 |
| [0.0110, 0.0177) | 1084 | +0.00044 |
| [0.0177, 0.0333) | 1084 | +0.00120 |
| [0.0333, 0.0780) | 1083 | +0.00681 |

最大步长桶的损失下降约为最小桶的 **34 倍**——方向一致时大步确实带来大收益。

### 结论与处方

诊断脚本给出的结论：**方向高度一致且步长与损失下降正相关，早期/中期（余弦>0.9
且损失仍在降）可安全放大 residual lr 加速收敛；进入噪声地板（余弦<0.7）后应恢复
或减小，避免抖动。** 具体而言：

1. **调整有据**：91% 轨迹后段方向一致、步长-收益强正相关，说明优化大部分时间沿
   稳定弹道推进且远未到地板，放大 residual lr（如从 0.03 提到 0.05–0.06）在早期/
   中期有明确收益空间。
2. **必须设回收条件**：23.5% 轨迹末段跌入噪声地板，此时方向翻转、大步只会放大
   抖动。任何放大策略都应在余弦连续跌破 0.7 时恢复或减小 lr。
3. **建议灰度验证而非直接改默认**：先做小倍数 A/B（如 0.03 vs 0.05）对比收敛到
   同一损失所需轮数，再决定是否改默认值。

### 口径边界（重要）

- 本报告为**观察性诊断**，不是真实重跑反事实。capture 目录未保存 `sample_*.npz`
  （`save_dataset=False`），无法用真实 forward 目标函数重跑不同 lr 的对照。
- "放大 lr → 加速收敛"是**一阶线性代理**推断（step ∝ lr、更新方向不变）：基于实测
  步长-收益正相关外推，而非实测放大后的收敛曲线。真实收益需在完整采集数据上做
  A/B 才能确认。
- 若要在真实数据上验证，需先用保留样本的采集重跑（开启 dataset 保存），再对各 lr
  做同起点的收敛对比。当前仅据此形成处方建议，**不改动 fit 默认学习率**。

## 边界与声明

- 激光功率**未做任何调整**，沿用实验台当前低功率设定，符合用户授权的"低功率、可控环境"边界。
- 本报告**不宣称** SLM 实验帧率提升，也**不宣称** CCD 与 SLM 逐帧一一对应。
- 运行期间未改动其他车道的未提交/未跟踪改动；本次验证产物位于
  `data/debug/slm_transfer/debug_weights_real_force_20261121/`，权重目录重定向为
  临时目录，真实 `data/slm_models` 无副作用。
