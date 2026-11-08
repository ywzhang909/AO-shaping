# 门控循环残差头对比：gated / attention vs memoryless（#239 groundwork）

<!-- provenance:start -->
> **生成脚本**: [`scripts/slm/generate_gated_recurrent_report_20261107.py`](../../../scripts/slm/generate_gated_recurrent_report_20261107.py)
> **复现命令**: `python scripts/slm/generate_gated_recurrent_report_20261107.py`
> **运行环境**: 离线渲染（实验数据 2026-11-07：单模屏 ladder pass1/pass2 与 r2 存档，phase_sign=−1）
> **说明**: #239 门控循环残差头 对比择优（verdict inconclusive；4 变体×3 种子）：输入为 data/debug/slm_transfer/gated_recurrent_20261107/results.json 存档产物，表格数字全部渲染时读取、零硬编码。报告本体不占用设备。
<!-- provenance:end -->

- **日期**：2026-11-07
- **设备声明**：本报告为离线渲染，未占用任何 SLM / 相机 / 激光设备，未导入驱动模块，未执行采集或训练；数据全部来自磁盘上已归档的结果文件。
- **数据源**：`data/debug/slm_transfer/gated_recurrent_20261107/results.json`（生成于 2026-11-07T12:32:46+00:00；其设备声明："Offline analysis only. No SLM, camera or laser was opened; no driver module was imported and no --execute path was used. All figures come from archived npz frames on disk."）
- **训练脚本**：`scripts/slm/gated_recurrent_transfer_20261107.py`（本报告只读其归档结果，不重跑训练）
- **生成脚本**：`scripts/slm/generate_gated_recurrent_report_20261107.py`

## 1. 研究问题与变体设置

研究问题：在冻结物理预测模型之上学习残差头时，把上一帧 CCD、相位与残差作为循环输入（门控/注意力）能否比 memoryless 残差头更快达到同等保真度，或改善方向相关。#239 把 self-attention 残差列为末位备选方案，本实验为其提供离线对比证据。

四变体共享优化器（AdamW）、学习率、权重衰减、序列级 batching、迭代预算与数据；输出层零初始化，迭代 0 等于冻结 base 预测。Zernike 符号取 `phase_sign=-1`（修正已知方向缺陷）。训练配置：iters=300、eval_every=10、seeds=[0, 1, 2]、lr=0.005、weight_decay=0.0001、hidden=64、feat=16、out_res=32、attn_k=3、phase_sign=-1、torch=2.14.0+cu132、numpy=2.4.4。

| 变体 | 历史输入 | 循环 | 参数量 |
|---|---|---|---|
| `memoryless` | 当前相位 + 当前 base 预测 | 无 | 103,552 |
| `concat` | 当前 + 上一帧 CCD/相位/残差（拼接） | 无（前馈） | 152,704 |
| `gated` | 当前 + 上一帧 CCD/相位/残差 | GRU 式门控 | 242,880 |
| `attention` | 当前 + 对前 3 帧的 dot-product 注意力 | 注意力 | 821,184 |

数据与划分（序列不跨训练/验证边界，首帧全零平场参考不参与上一帧配对）：

| 域 | 可用帧 | 训练 / 验证 | 序列（训/验） | 核心像素 |
|---|---|---|---|---|
| r2 | 36 | 18 / 18 | 12 / 12 | 55 |
| ladder | 84 | 42 / 42 | 3 / 3 | 32 |

## 2. 两域 held-out 终值

± 为 3 种子间的标准差；base 行取 `base.minus.*`（冻结物理预测模型、无残差头，无 std）。

### r2 held-out

| 变体 | 参数量 | full Pearson | core Pearson | rel L2 |
|---|---|---|---|---|
| frozen base（无残差头） | — | 0.6473 | 0.4249 | 0.7849 |
| memoryless | 103,552 | 0.6940 ± 0.0259 | 0.1299 ± 0.0189 | 0.8291 ± 0.0113 |
| concat | 152,704 | 0.7059 ± 0.0062 | 0.1146 ± 0.0039 | 0.8144 ± 0.0103 |
| gated | 242,880 | 0.7037 ± 0.0028 | 0.1623 ± 0.0327 | 0.8832 ± 0.0078 |
| attention | 821,184 | 0.6669 ± 0.0505 | 0.1386 ± 0.0187 | 2.2600 ± 2.2071 |

### ladder held-out（pass1 训练 / pass2 验证）

| 变体 | 参数量 | full Pearson | core Pearson | rel L2 |
|---|---|---|---|---|
| frozen base（无残差头） | — | 0.8418 | 0.7008 | 0.5363 |
| memoryless | 103,552 | 0.9395 ± 0.0030 | 0.4639 ± 0.0124 | 0.3405 ± 0.0083 |
| concat | 152,704 | 0.8902 ± 0.0150 | 0.5076 ± 0.0254 | 0.6252 ± 0.0538 |
| gated | 242,880 | 0.9380 ± 0.0016 | 0.4452 ± 0.0068 | 0.3409 ± 0.0044 |
| attention | 821,184 | 0.8370 ± 0.0134 | 0.4102 ± 0.0157 | 0.8793 ± 0.0815 |

## 3. 收敛阈值迭代数

评估粒度 10 次迭代；单元格为达到阈值的平均迭代数 ± 种子间 std（达成种子数/总种子数）。

### r2 域

| 变体 | 参数量 | r ≥ 0.66 | r ≥ 0.68 | r ≥ 0.7 |
|---|---|---|---|---|
| memoryless | 103,552 | 13 ± 6（3/3） | 17 ± 6（3/3） | 27 ± 15（3/3） |
| concat | 152,704 | 13 ± 6（3/3） | 20 ± 0（3/3） | 30 ± 17（3/3） |
| gated | 242,880 | 10 ± 0（3/3） | 13 ± 6（3/3） | 27 ± 6（3/3） |
| attention | 821,184 | 73 ± 55（3/3） | 127 ± 65（3/3） | 210 ± 0（1/3） |

### ladder 域

| 变体 | 参数量 | r ≥ 0.86 | r ≥ 0.88 | r ≥ 0.9 |
|---|---|---|---|---|
| memoryless | 103,552 | 10 ± 0（3/3） | 10 ± 0（3/3） | 20 ± 0（3/3） |
| concat | 152,704 | 17 ± 12（3/3） | 23 ± 12（3/3） | 37 ± 12（3/3） |
| gated | 242,880 | 13 ± 6（3/3） | 17 ± 6（3/3） | 20 ± 0（3/3） |
| attention | 821,184 | 30 ± 0（1/3） | 30 ± 0（1/3） | 30 ± 0（1/3） |

## 4. 方向均值（42 对）

冻结 base 为 42 对 ±相位指令上逐对方向 Pearson 的均值 ± 逐对 std；各变体为 3 种子最终模型的方向均值 ± 种子间 std（每个变体每种子各 42 对）。

| 变体 | 参数量 | 方向 Pearson 均值 ± std | 种子达成 | 评估对数 |
|---|---|---|---|---|
| frozen base | — | 0.5596 ± 0.2020 | —（逐对统计） | 42 |
| memoryless | 103,552 | 0.7598 ± 0.0057 | 3/3 | 42 |
| concat | 152,704 | 0.6490 ± 0.0530 | 3/3 | 42 |
| gated | 242,880 | 0.6297 ± 0.0214 | 3/3 | 42 |
| attention | 821,184 | 0.4756 ± 0.0478 | 3/3 | 42 |

## 5. Verdict 与解读

- **判定**（results.json `verdict` 字段）：r2 与 ladder 均为 **inconclusive**（winner: none；wins 0 / losses 0 / ties 3）——门控变体与 memoryless 在全部收敛阈值上的迭代数差异都在 10 次评估粒度内。
- **r2 域**：0.66：gated 10 vs memoryless 13；0.68：gated 13 vs memoryless 17；0.7：gated 27 vs memoryless 27。门控在两个较低阈值约快 3 次评估，最高阈值持平——差异恰在一个评估粒度内。
- **ladder 域**：0.86：gated 13 vs memoryless 10；0.88：gated 17 vs memoryless 10；0.9：gated 20 vs memoryless 20。门控持平或更慢；memoryless 以 103,552 参数达到相当终值（0.9395 ± 0.0030 vs 0.9380 ± 0.0016）且更早达标。
- **两域合看**：门控没有观察到一致的迭代数压缩；附加的循环输入未换来收敛效率。
- **attention 变体崩溃**：r2 域 rel L2 2.26 ± 2.21、最高阈值仅 1/3 种子达成（210 次迭代）；ladder 域全部阈值仅 1/3 种子达成。在 821,184 参数、无结构约束的前提下不构成可保留的变体——与 #239 将 attention 列为"末位备选、gate 零初始化默认关"的定位一致。
- **方向指标**：memoryless 把方向均值从 0.5596（frozen base）抬到 0.7598——方向指标可以通过学习残差头改善，而不必先解决物理预测模型；但这是代理指标，不等于闭环收敛，方向门验收仍需独立验证探针。

## 6. Caveats

1. **held-out 测的是复现性而非泛化**：ladder pass2 重复 pass1 的同一批 ±指令，验证集相位族与训练集不相交这一口径在本实验中未满足。
2. **统计功效有限**：n=3 种子，阈值迭代数以 10 次评估为粒度记录，±3 次评估的差异不可分辨。
3. **`phase_sign=-1` 的代价**：修正方向的同时压低绝对匹配——r2 base 0.6473（+1 时 0.7984）、ladder_holdout（OOD）base 0.6345（+1 时 0.8345）；本文所有变体表均为 −1 口径。
4. **方向均值的改善是代理指标**：memoryless 把方向均值 0.560 → 0.760，说明该指标可由残差头改善，但不等于闭环收敛；不能据此跳过 #239 的方向门与结构项前置条件。

## 7. 复现命令与产物

```bash
# 渲染本报告（离线，无设备占用，不重跑训练）
python scripts/slm/generate_gated_recurrent_report_20261107.py
```

训练数据由 `scripts/slm/gated_recurrent_transfer_20261107.py` 生成并已归档，勿在离线渲染流程中重跑。产物路径：

- `data/debug/slm_transfer/gated_recurrent_20261107/results.json` —— 全部表格数字的唯一来源
- `data/debug/slm_transfer/gated_recurrent_20261107/REPORT.md` —— 训练侧原始报告
- `data/debug/slm_transfer/gated_recurrent_20261107/weights.pt` —— 各变体权重
- `report/shape_trials/gated_recurrent_20261107/gated_recurrent_20261107.md` —— 本报告
