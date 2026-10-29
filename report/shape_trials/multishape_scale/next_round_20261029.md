# 2026-10-29 之后的下一步：质量改善实施方案

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`src/ao_shaping/tools/slm/transfer/jacobian_guard.py`](../../../src/ao_shaping/tools/slm/transfer/jacobian_guard.py)
> **数据/关联脚本**: [`src/ao_shaping/algorithm/signal_processing/spectral_regulariser.py`](../../../src/ao_shaping/algorithm/signal_processing/spectral_regulariser.py)
> **数据/关联脚本**: [`src/ao_shaping/algorithm/signal_processing/differentiable_defocus_uniformity.py`](../../../src/ao_shaping/algorithm/signal_processing/differentiable_defocus_uniformity.py)
> **数据/关联脚本**: [`src/ao_shaping/tools/slm/transfer/validate.py`](../../../src/ao_shaping/tools/slm/transfer/validate.py)
> **数据/关联脚本**: [`src/ao_shaping/runners/slm/transfer_shapes_runner.py`](../../../src/ao_shaping/runners/slm/transfer_shapes_runner.py)
> **数据/关联脚本**: [`scripts/slm/audit_replay_blocks.py`](../../../scripts/slm/audit_replay_blocks.py)
> **运行环境**: 离线（issue #213 质量建议 + 2026-10-29 会话实测记录；本文的代码项已在仓内实现并通过单元测试，尚未实验）
> **说明**: 2026-10-29 之后的质量改善实施方案：按 issue #213 comment 6466195110 的质量侧优先级（3a Jacobian 逐块校验 / 1b 频域 MTF 正则 / 2b 像素级不均匀性掩膜 / 4c 多目标 Pareto）给出改法、开关、验收判据与失败形态；依据本轮实测把「饱和先决」与「审计 clipped 语义修正」提到 P0，并明确 3a 只实现缩 cap（不转纯实测梯度）、1b 默认关闭等实测 MTF。所有判定统一使用重复导出的不确定度（6 个 ABBA 周期可达 p<0.05），不用固定阈值。§9 落地状态区分已实现（3a / 1b / P0-2 审计 / 4c 阶段化权重）与未实现（2b 像素级掩膜 —— 缺实测标定数据，反演无法验证，理由写在 §5 末尾），并记下本轮走 `cv` 变体导致形状覆盖 / 边界误差未测这一字段缺口。
<!-- provenance:end -->

> **来源**：[issue #213 comment 6466195110](https://github.com/ywzhang909/AO-shaping/issues/213#issuecomment-6466195110)
> 的质量侧优先级表，以及同日实验会话暴露的三个具体缺陷
> （[live_20261029_device_session.md](live_20261029_device_session.md)）。
>
> **本文不是又一次"提几个想法"**：每一项都给出**改哪个开关**、**怎么判成立/不成立**、
> **失败长什么样**。已实现的两项（3a、1b）代码在仓内并通过单元测试，尚未实验。

---

## 0. 优先级重排（与 issue 表的差异及理由）

issue 表的顺序是 `3a → 1b → 2b → 4c`，理由是"3a 复用本块已有的 A1/B 帧，几乎免费"。
这个理由成立，但本轮实验的**实测**给出两条更前置的事实：

> **一、测量先坏了一半。** 200 px 上四个形状里三个出现满量程截断
> （circle 0 / triangle 2 / square 4 / star 10，各 15 行）。star 剔除后只剩 5 行、
> 一档可判；triangle 第三档失去 seed 行。
>
> **二、square 的"变差"不是测量问题。** 剔除饱和行之后，两个可用曝光档的
> `dCV` 仍为 `+0.1704 / +0.1646`，而噪声半宽只有 `0.0526 / 0.0547`——效应量约为
> 噪声的三倍。闭环确实把它做差了，这指向**模型步方向**，也正是 3a 的目标。

所以在读"模型方向准不准"之前，必须先把饱和行从统计里真正剔除（否则结论会
被截断画面污染），但 3a 本身不需要等——它是代码改动，与饱和是两件事。

| 序 | 项 | 类型 | 状态 |
|---|---|---|---|
| **P0-0** | 饱和先决：重放档自动压到 `saturated_fraction=0` | 运行参数 | **必须先做** |
| **P0-1** | `3a` 逐块 Jacobian 方向校验 + 自适应信任域 | 代码 | **已实现**（默认开） |
| **P0-2** | 审计脚本 `clipped` 语义修正（标了却没真正排除） | 代码 | **已实现**（见 §3） |
| **P1** | `1b` 频域 MTF-matched 正则 | 代码 | **已实现**（默认关） |
| **P2** | `2b` SLM 像素级不均匀性掩膜 | 代码 + 一次标定 | 待实现 |
| **P3** | `4c` 多目标 Pareto 动态权重 | 代码 + 验收口径重钉 | 待实现 |

---

## 1. P0-0 饱和先决（没有它，后面三项都白测）

### 问题

重放用 `--replay-exposures auto` 按目标自动选档，但 auto 只保证"不整体过曝"，
不保证"目标框内不截断"。200 px 目标的能量密度高于 1200 px，同一档曝光在
200 px 上直接顶到满量程。

### 改法

1. 重放前对每个候选档先采一次，读 `saturated_fraction`（审计里已有这个量）；
2. 任何档 `saturated_fraction > 0` 就整体降一档重来，直到全部为 0；
3. 若降到相机下限仍不为 0，**该组合不产出结论**，在 journal 里记
   `status=saturated_out`，并在报告里明确写"测量无效"，不得用未饱和档之外的
   数据拼凑。

### 验收

- 下一轮所有报告的饱和行数为 **0**；
- 若某组合必须标 `saturated_out`，报告里该组合的三口径栏写 `—`，不是写数字。

### 失败长什么样

降档后 `ee_rel` 骤降而 `cv` 改善——那是曝光不足导致的本底占比升高，不是整形
变好。判据见 §1.1。

### 1.1 三口径必须同时报（不可只报 CV）

| 口径 | 答什么问题 | 单独读会怎样错 |
|---|---|---|
| 冻结核心 `cv` | 目标区内是否更均匀 | 忽略"目标框外漏光" |
| 形状覆盖 / 边界误差 | 光斑是否真的长成目标形状 | 圆形目标下 CV 天然偏低 |
| 相对 `ee_rel` | 能量是否还在目标里 | 忽略"均匀但极暗" |
| `zero_excluded_fraction` | 零级是否被排除在统计外 | 漏报时 CV 被零级污染 |
| 饱和比例 | 读数是否可信 | 饱和行的 CV 无意义 |

任一组合缺其中一项，该组合**不计入结论**。

---

## 2. P0-1 `3a` 逐块 Jacobian 校验 + 自适应信任域（已实现）

### 它治的是什么

issue 表的天花板 ②：锚定模式假设冻结 Jacobian 的方向大致正确。既有实测里出现过
**重拟合 loss 0.93 → 0.36，而配对 Δ ≈ 0**——模型在变准，实测没变。方向错了，
步长再小也只是沿着错的方向慢慢走。

### 实现

`src/ao_shaping/tools/slm/transfer/jacobian_guard.py`

- `direction_agreement(anchor_pred, cand_pred, anchor_meas, cand_meas)`：把两对
  画面的**差分**做零均值化后求相关系数，同时给出尺度无关的相对 L2。
- `TrustRegionSchedule(cap, floor, low, high, shrink, grow)`：`agree` 时
  `cap × grow`（上限回到流程表给的值），`contradict` 时 `cap × shrink`
  （下限 `floor`），`weak` / `insufficient_signal` 不动。
- **只调度下一块的 `phase_rms_cap`，从不参与接受/拒绝**。这一点是刻意的：
  接受/拒绝已有自己的判据，再加一层会让"为什么这一块被拒"变得不可解释。

接线在 `validate.py` 的 `ccd_check` 分支：用本块已有的 A1 锚定帧与 B 帧，
不额外采集；结果写进 `decision["jacobian"] = {correlation, relative_l2,
verdict, cap_next}`。

### 开关

| 开关 | 默认 | 含义 |
|---|---|---|
| `--jacobian-guard / --no-jacobian-guard` | 开启 | 只在 `--ccd-check` 生效；无 CCD 校验的阶段本来就没有实测差分可对 |
| `--jacobian-low` | `0.5` | 低于此判 `contradict` |
| `--jacobian-high` | `0.85` | 高于此判 `agree` |
| `--jacobian-floor` | `0.02` | cap 下限（rad，照明加权 RMS） |

### 验收（下一轮实验）

0. **阈值先在孪生上标定**：`--jacobian-low/high` 必须先在
   `drivers/sim/bench_twin.py` 的数字孪生上，用**已知真值方向**的相位增量确认
   两个阈值能区分「方向对」与「方向错」。**不拿实验数据调阈值**——那等于用
   结论去证明结论。
1. **可解释性**：每个 `ccd_check=True` 的块都有 `decision["jacobian"]`，
   四个字段齐全；缺失即为接线回归。
2. **有效性**：在 square / star 这两个本轮"方向反了/没差别"的形状上，
   `verdict` 的分布应当是 `contradict` 占多数。若全为 `agree`，说明阈值或
   差分定义不对，**不是**模型变好了。
3. **无害性**：全轮结束时，被接受块的比例与 `--no-jacobian-guard` 同配置对照
   相比不显著下降（guard 只缩步长，不该把块逼成全拒）。对照必须同
   `bit_depth` / `window_step`，否则不可比。
4. **收益**：同形状、同种子、guard 开/关各 6 次重复，配对比较最终 CV；
   用重复导出的不确定度判定（见 §6），**不用固定阈值**。

> **与质量评论的差异（明确记录）**：评论给的是「缩 cap **或**转纯实测梯度」，
> 本轮**只实现缩 cap**。理由：锚定模式的 delta 本来就是实测差分驱动的，
> 「转纯实测梯度」会同时换掉目标函数与步长，两个变量一起动就无法归因。
> 若上面的第 4 条不通过，再把「纯实测梯度」作为**单独**的下一个实验项。

### 失败长什么样

`cap` 一路掉到 `floor` 且再也不涨——差分被噪声主导（`insufficient_signal`
被误判为 `contradict`，或相关窗口选在零级附近）。处置：先看
`insufficient_signal` 的计数，若为 0 而 `contradict` 满格，就是阈值定得太高。

---

## 3. P0-2 审计脚本的 `clipped` 语义（**已实现**）

### 问题

`scripts/slm/audit_replay_blocks.py` 打印

```
! clipped (>= 98% full scale), excluded from any comparison: <labels>
```

但 `usable` 只排除 `error` 行，`clipped` 行**仍进入分组均值**，`n_usable` 也是
排除 error 后的数。也就是说：**声称排除，实际没排除**。

### 改法（已落地）

```python
measured = [r for r in rows if not r.get("error")]      # 无 error
clipped  = [r["label"] for r in measured if r.get("clipped")]
usable   = [r for r in measured if not r.get("clipped")] # 判定只用这个
...
for key, members in group_rows(usable).items():
```

返回字典给三个计数，避免歧义：

```
n_rows     总行数
n_measured 无 error
n_usable   无 error 且未饱和  ← 判定只用这个
```

打印同时改了三处：表头改为「`n_usable/n_measured` measured rows usable,
`n_rows` present」，clip 行改为「dropped from every group below」，并且
分组为空时显式打印「no comparable groups」而不是静默输出空表。

### 验收结果（对本轮 s200 四形状重跑）

| 形状 | 修正前 | 修正后 | 变化 |
|---|---|---|---|
| circle | 15/15 | 15/15，`dCV` 一位未变 | 天然回归基准，通过 |
| triangle | 2/15 饱和，三档判定 | 13/15，**两档**判定 | 第三档 seed 行被截断 → 该档消失 |
| square | 4/15 饱和，三档判定 | 11/15，**两档**判定 | 第三档 4 个 final 全截断 → 该档消失 |
| star | 10/15 饱和，三档判定 | 5/15，**一档**判定 | 后两档整组截断 → 两个判定撤回 |

**关键结论**：被剔除的饱和行集中在**最高曝光档**，所以低两档的 `dCV` 数值没有变
（square 仍是 `+0.1704 / +0.1646`，triangle 仍是 `−0.1613 / −0.1732`）。变的
是「哪些档能判」。因此 square 的 `wrong_direction` **不是**饱和假象，而是真实回归。

`n_usable` 与各分组 `n` 之和一致（circle 5+5+5=15，triangle 5+5+3=13，
square 5+5+1=11，star 5=5）。

### 为什么排在 1b 之前

1b 的收益要靠审计来判；审计本身在骗人时，1b 的"有效/无效"无法区分。

---

## 4. P1 `1b` 频域 MTF-matched 正则（已实现，默认关）

### 它治的是什么

issue 表的天花板 ①：控制网格上存在 Nyquist 上界，模型认为"有效"的高频相位
器件渲染不出来，会在远场走叠成散斑，而散斑反过来污染 CV，闭环无法纠正。
现有的空域 `1 − cos` 平滑项只在**空域**抑制相邻像素差，与器件实际能渲染的
频带没有直接对应。

### 实现

`src/ao_shaping/algorithm/signal_processing/spectral_regulariser.py`

- `mtf_matched_mask(grid, cutoff, rolloff)`：径向通过带，`cutoff + rolloff ≤ 0.5`；
  过渡带内线性下降；`rolloff=0` 退化为硬截断。
- `spectral_penalty(delta, mask)`：`Σ|FFT(delta)·(1−mask)|² / Σ|FFT(delta)|²`，
  即**带外功率占比**。归一化使它对整体幅度不变——它约束的是"能量往哪个频段去"，
  不是"幅度小不小"。
- `control_grid_cutoff(grid)`：Nyquist 的 70%（`0.35` cycles/控制像素），
  过渡带 0.10。

接线在 `DifferentiableUniformityOptimizer.update()`，不是 `ShapeObjective`：
锚定模式下 objective 收到的差分已 detach，把惩罚放在 objective 里拿不到梯度；
实测模式则单独 `penalty.backward()` 累加进 `delta.grad`。

### 开关

| 开关 | 默认 | 含义 |
|---|---|---|
| `--spectral-weight` | `0.0`（**关闭**） | 惩罚权重 |
| `--spectral-cutoff` | 未指定 → `0.35` | 通过带边缘（cycles/控制像素） |

默认关闭是刻意的：**器件 MTF 还没实测**。在没有 MTF 的情况下开一个"MTF-matched"
正则，名字与实质不符。

### 前置：怎么在没有 MTF 时也能用

两条路，按成本排序：

1. **扫描式替代门禁（下一轮就能做，不需要新设备）**：固定 weight，扫
   `cutoff ∈ {0.25, 0.30, 0.35, 0.40, 0.45}`，每个取值 6 次重复。若最优 cutoff
   落在 0.45（接近全通），说明当前控制网格根本不是瓶颈，1b 应当关掉；若落在
   0.25–0.30 且 CV 显著优于 0.35，说明器件带宽确实不足，再用那个值。
2. **实测 MTF（需要一次专门标定）**：在控制网格上投影单频条纹，测远场对比度
   随频率的衰减，拟合出 `cutoff`。这是唯一能让"MTF-matched"名副其实的做法。

### 验收

1. **带限有效性**：开启后最终相位的带外功率占比必须低于关闭时的
   （同一形状、同一种子、同一块数）；审计应当能看到这个差。
2. **不削边缘**：形状覆盖 / 边界误差不得显著变差。若覆盖率下降而 CV 改善，
   那是把光斑磨圆了——退化解，判为不成立。
3. **重复性**：≥6 次重复，配对比较；p<0.05 才写"成立"，否则只报效应量与
   置信区间。

### 失败长什么样

`cutoff` 过低（<0.2）时目标边缘变糊、`ee_rel` 下降但 `cv` 好看。这是典型的
"把目标改成高斯"的退化解，必须靠 §1.1 的形状覆盖口径挡住。

---

## 5. P2 `2b` 像素级不均匀性掩膜（待实现）

### 它治的是什么

`amplitude_residual` 只建模低频照明残余；SLM 面板像素级的增益/相位偏移没有
进物理预测模型。250 px 级别的目标上，这类偏差会直接变成 Jacobian 的方向误差
（即 3a 想用相关性探测的东西）。

### 实现要点

1. 一次标定：在控制网格上扫一组低频 Walsh/Hadamard 基，从远场反演每个控制
   像素的复增益；产出 `gain_map` + `phase_offset_map`（控制网格分辨率）。
2. 乘入前向：`forward` 的 SLM 复振幅乘以 `gain_map · exp(i·phase_offset_map)`
   后再传播。**不改**优化器结构。
3. 参数化必须远少于数据点：`gain_map`/`phase_offset_map` 各用一个低阶
   （≤ 8×8）平滑基表示、再插值到控制网格，避免过参数化伤拟合。

### 前置门禁（issue 表点名的那条）

**必须过验证集泛化门禁**：标定用的相位族与开环评估用的相位族不相交；
模型在验证集上的 loss 下降幅度要与训练集同量级。若训练 loss 降、验证 loss
不降，判为过拟合，退回不加掩膜。

### 验收

- 验证集 loss 改善 ≥ 训练集改善的 50%；
- 开环预测量子与实测的配对 Δ 显著小于无掩膜时的（这是"提 Jacobian 保真度"的
  直接检验）；
- 标定图时效性：隔 24 h 重测，`gain_map` 的相对漂移 < 5%。

### 为什么本轮没实现（诚实记录）

`2b` 的核心不是"加一个 mask 字段"，而是**一次真实标定**加**一次验证集泛化判定**：
Walsh/Hadamard 扫基、从远场反演每像素复增益、用不相交相位族做训练/验证划分。
设备已下线，仓里没有任何一份这样的标定数据，于是：

- 反演代码写了也无法执行——符号约定、归一化、基函数顺序全都没有数据可对；
- 而"写了就算实现"最坏的结果是下一轮把它当既有件直接放进闭环，用一个未验证的
  增益图去乘前向振幅，方向错了只会让闭环更差（这正是 3a 要探测的那类失真）。

所以 2b 的处理是：**设计留在本节，代码等标定数据**。真要做时按下面顺序——
先跑一次标定采集并落盘原始远场，再写反演并用**验证集的相位族**验证，验证集不达标
就退回不加 mask（本节的验收条款已经是这个意思）。

---

## 6. P3 `4c` 多目标 Pareto 动态权重（已实现，默认关）

### 它治的是什么

纯 CV 会收敛到"均匀但极暗"或"均匀但零级没抑"的解。objective 现在已有
`cv2 + energy + shape + smooth`，但权重是常数，不随进程变化。

### 实现要点

- 阶段化权重：前 1/3 块偏 `energy`（先把光放进目标框），中段偏 `shape`
  （长成目标形状），后 1/3 偏 `cv2`（磨均匀度）。
- 权重进 `plan` 与每个块的 `decision`，否则事后无法复盘"这一块为什么这么走"。

### 前置（issue 表点名的风险）

**改变验收口径需同步重钉验收判据**。权重 schedule 只改变优化路径，不改变
判据；但如果为了让 CV 好看而调 `energy` 权重，那就等于改判据。规则：

> 验收判据（§1.1 五项）在一次实验内**冻结**；权重 schedule 只能作为
> 优化过程的可调项。任何"为了让结果通过而改判据"的改动，必须在报告里
> 单列并标注为口径变更。

### 验收

- 三个目标的达成度（CV 改善 / `ee_rel` 不降 / 零级抑制度不降）同时成立，
  不是"CV 独好"；
- 与常数权重对照 ≥6 次重复配对比较。

### 落地（2026-10-29，默认关）

| 件 | 位置 |
|---|---|
| 权重变成具名字典（原来是散在 `evaluate` 里的常数，行为逐字节不变） | `algorithm/signal_processing/multishape_objective.py::DEFAULT_WEIGHTS` + `set_weights()` |
| 三阶段 schedule（前 1/3 偏 `energy`、中段偏 `shape`、后 1/3 偏 `cv2`） | `algorithm/signal_processing/weight_schedule.py` |
| 逐块应用 + 把**实际用到的权重**写进 `summary.json` 条目 | `tools/slm/transfer/validate.py`（`entry["weights"]`） |
| CLI 开关 | `--weight-schedule {constant,staged}`，以及它的前置 `--objective {cv,shape,formation}` |

三点实施口径，避免下一轮误读：

1. **`cv` 变体不可 staged**：CV-only 损失只有 `cv2 + smooth`，没有 `energy`/`shape`
   可调，`staged` 在那里等于空转。所以 `--weight-schedule staged` 与
   `--objective cv` 的组合在**解析参数时**就被拒绝（`click.UsageError`），而不是
   静默跑一个没生效的 schedule。
2. **`--objective` 本身是"换变体"，不是开关**：本轮全部数据走 `cv` 变体
   （`cv_only=True`，见 `live_20261029_device_session.md` §5.5），换成 `shape` /
   `formation` 后 loss 组成不同，结果不能与 `cv` 变体的 dCV 混读。`formation` 还是
   补上形状覆盖 / 边界误差两项（`l_partition` / `l_boundary`）的唯一途径。
3. **判据仍然冻结**：`accept_metric` 依旧是实测 `cv`，schedule 只改优化路径；
   如果哪天为了让 CV 好看而改 `energy` 权重，那是改判据，必须单列。

单测：`tests/ao_shaping/algorithm/test_weight_schedule.py`（14 项：三阶段切分、
只缩放三键、formation 专有项不动、`cv` 变体退化、未知 mode/键报错、默认权重逐字节
复现历史 loss、staged 确实改变同一帧的 loss）。

---

## 7. 下一轮 lane 编排

沿用本轮目录约定（一会话一形状、`<session>/<shape>/common_replay`），
但**先跑饱和先决**：

| 批 | 内容 | 条数 | 目的 |
|---|---|---|---|
| A | 饱和标定：四形状 × 三尺度，只读 `saturated_fraction` | 12 | 定出每个组合的可用曝光上限 |
| B | `analytic_init_s200` 四形状，guard 开 | 4 | 复现本轮，验证 P0-2 修正 |
| C | `analytic_init_s200` 四形状，guard 关（对照） | 4 | 3a 的配对对照 |
| D | `analytic_init_s200` 四形状，spectral cutoff 扫描 | 20 | 1b 的替代门禁（§4） |
| E | `analytic_init_s600` 四形状，guard 开 | 4 | 补本轮未完成的尺度 |
| F | `strict_cold_s200` 四形状 | 4 | **首次**回答两类初始化的差异 |

共 48 条。**E、F 两批只在 A–D 全部给出干净读数后才开跑**——否则又是在
饱和数据上判尺度。

每条 lane 的收尾固定为：`common_replay` → `audit_replay_blocks.py
--session <session>/<shape>` → 把三口径 + 五项审计写进当次 live 日志。

---

## 8. 每项成立的判定（统一）

| 项 | 判据 | 数据量 | 不成立时怎么写 |
|---|---|---|---|
| P0-0 饱和 | 饱和行数 = 0 | 全部 | 该组合三口径写 `—`，写"测量无效" |
| P0-1 3a | `contradict` 集中在 known-bad 形状；接受率不显著下降；配对 CV 改善 | 6 次重复/配置 | 只报效应量与 CI，不写"成立" |
| P0-2 审计 | circle 数字一位不变；square 数字变化 | 本轮四形状 | 工具回归，不进结论 |
| P1 1b | 带外功率下降 + 覆盖率不降 + 配对 p<0.05 | 6 次重复/cutoff | 写"未观察到收益"，不写"无效" |
| P2 2b | 验证集改善 ≥ 训练集 50% + 开环配对 Δ 变小 | 独立验证族 | 判过拟合，退回 |
| P3 4c | 三目标同时成立 + 配对比较 | 6 次重复 | 写"仅 CV 改善"，判退化解 |

**统一红线**：任何一项的结论都必须由**重复导出的不确定度**支撑，不得用固定
阈值（如"CV 降 5% 算成功"）。4 次重复在结构上到不了 p<0.05，只报效应量。

---

## 9. 落地状态

| 项 | 代码 | 测试 | 实验 |
|---|---|---|---|
| P0-1 3a | `tools/slm/transfer/jacobian_guard.py` + `validate.py` 接线 | `test_jacobian_guard.py` 10 项 | 待跑 |
| P1 1b | `algorithm/signal_processing/spectral_regulariser.py` + `differentiable_defocus_uniformity.py` | `test_spectral_regulariser.py` 9 项 | 待跑（默认关） |
| P0-2 审计 `clipped` 语义 | **已实现**：`scripts/slm/audit_replay_blocks.py` 分 `measured` / `clipped` / `usable` 三集，分组只用 `usable`；无可用组直接 `no comparable groups` | 已重跑四形状验证 | 已生效 |
| P0-0 饱和 | 运行参数 + 报告约定 | — | 待跑 |
| 五项口径的「形状覆盖 / 边界误差」 | 不需新代码：`ShapeObjective(formation=True)` 的 `l_partition` / `l_boundary` 已是 canonical 实现 | — | 下一轮随每块 `decision` 落盘（`--objective formation` 已接线）；换变体即换 loss，须单独成组、不与 `cv` 变体混读 |
| P2 2b | **未实现**，理由见 §5 末尾：需要一次真实标定 + 验证集泛化门禁，没有实测数据时写出来的反演无法验证 | — | — |
| P3 4c | **已实现**（默认关）：`weight_schedule.py` + `ShapeObjective.set_weights` + 逐块 `entry["weights"]`；CLI `--weight-schedule` 与其前置 `--objective` | `test_weight_schedule.py` 14 项 + runner 3 项 | 待跑（`constant` + `cv` = 历史行为） |

**本轮数据的字段缺口**（台账见 `live_20261029_device_session.md` §5.5）：本轮
objective 走 `cv` 变体（`--objective cv` ⇒ `cv_only=True`），抽查 `circle/mono12_cv`
第 0 轮的 `iterations.pkl`，`l_partition` / `l_boundary` / `l_peripheral` /
`l_floor` **逐轮恒为 0.0** —— 形状覆盖 / 边界误差是**没有测**，不是没报。按 §1.1，本轮结论只覆盖
「更均匀 + 能量没丢」两项；star 的 `ee_rel = 0.360` 也正落在这个空口上。三条补齐
方式（形状项随 `decision` 落盘、每阶段一次 Noll 序 Zernike 拟合、每块
predicted-vs-measured 标量差）写在 `live_20261029_device_session.md` §5.5。

P0-0 / P2 的具体代码在下一轮开工时按本文件实施；P0-1 / P1 / P0-2 / P3
已随本轮提交，开关默认值按上表设定（3a 与 `--objective cv` 默认开启/维持历史行为，1b 与 4c 默认关闭）。
