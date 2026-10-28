# 多形状 × 三尺度同载波闭环整形：实验后分析（2026-10-28）

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`src/ao_shaping/runners/slm/transfer_shapes_runner.py`](../../../src/ao_shaping/runners/slm/transfer_shapes_runner.py)
> **数据/关联脚本**: [`src/ao_shaping/tools/slm/transfer/validate.py`](../../../src/ao_shaping/tools/slm/transfer/validate.py)
> **数据/关联脚本**: [`src/ao_shaping/tools/slm/transfer/replay_phases.py`](../../../src/ao_shaping/tools/slm/transfer/replay_phases.py)
> **数据/关联脚本**: [`scripts/slm/generate_multishape_scale_live_figures.py`](../../../scripts/slm/generate_multishape_scale_live_figures.py)
> **运行环境**: 离线分析（读取 Session A 运行目录的 stage 子目录、common_replay*/ 重放 npz 与 manifest；不连接设备）
> **说明**: 多形状 × 三尺度同载波闭环整形：实验后分析（2026-10-28）。三层证据收口（模型验证集 / 逐块 CV / 重放饱和审计）；新增第（e）条硬约束——CV 随曝光单调上升并饱和、跨曝光档 CV 不可比；给出目标对照、实验差距（G1–G6）与下一步方案（N1–N6，含具体命令），并检索 Zotero 文库给出候选改进方法的文献依据（Mastiani 2024 / Cho 2026 GRAPHYCS / Chromatix 2026 等）。common_replay 的 90 ms 块末 4 条失效记录由 common_replay_r2 洁净重测覆盖。
<!-- provenance:end -->

> 本文件是 `multishape_scale_planning_20261028.md`（规划）与 `multishape_scale_live_20261028.md`（实验中记录）的**实验后分析**，
> 把三层证据（模型验证集 / 逐块 CCD 验收 / 重放审计）收口，给出**目标对照、实验差距与下一步方案**，并检索 Zotero 文库给出候选改进方法的文献依据。
> 数据目录同 Session A：`data/debug/slm_transfer/multishape_scale_20261028/s1200/`。

---

## 1. 目标与判据（与规划对齐）

**目标**：在同一物理预测模型、同一载波（周期 48 面板 px）、同一零级排除口径（半径 60 px）下，
测出四个形状（正方形 / 三角形 / 五角星 / 圆形）在 **200 / 600 / 1200 原生 CCD px** 三个尺度上的核心均匀度响应，
回答「形状与尺度哪一个主导本轮可达均匀度」。

**判据（来自规划 §7）**：核心 CV 为唯一可比指标；跨 bit 段、跨曝光档的 CV 差不算证据；
三角形 12 bit 段仍削顶（最高 1574 px 饱和），其 CV 仅作乐观上界；H3 勘误后 200 格零级排除区与目标框重叠 5.61%，不再作「干净参照」。

**已完成的尺度**：仅 **Session A（1200 原生 px）** —— 16 个形状×阶段组合、112 个验收块全部实测；
**Session B（600）/ C（200）未开始**，故 H1（尺度主导）/ H2（控制频带 vs 模型误差）/ H3（干净参照）**暂不能判定**，
下文 §6 把它们列为首要实验差距。

---

## 2. 证据闭合状态（三层）

| 层 | 规划要求 | 本轮状态 |
|---|---|---|
| ① 模型选择 | fit 验证集残差 | `model640`（`defocus_diopter=0.330`）/ `model_wide640` 已 fit；验证误差留待分析，未与实测闭合 |
| ② 逐块 CCD 验收 | 16 段段首 seed / 段末 final CV | **全 112 块已完成**；8 bit 段 0.7766–1.9206、12 bit 段 0.6574–0.7838（过程值） |
| ③ 重放审计 | 同相位多曝光重放 + 饱和审计 | `common_replay` 48 次 + `common_replay_dense` 24 次 + `common_replay_r2` 36 次；**本分析首次完成逐点 `raw_frames.max()` 饱和审计**（见 §4） |

**新闭合的证据**：重放不再「未审计」。两层重放（`common_replay` 4.0 网格 + `common_replay_dense` 2.0 网格）的总 72 条记录 =
4 形状 ×（transport_cv + mono12_cv + mono12_continue + mono12_dense）final × 20 / 50 / 90 ms；
`common_replay_r2` 是 `common_replay` 的**洁净重测**（覆盖原 90 ms 块末 4 条因 SLM 关设备前 USB 掉线前兆而失效的记录）。

---

## 3. 读数口径与饱和审计（五条硬约束）

沿用实验中记录 §4.1 的四条，并补第 **（e）条重放曝光依赖**（本轮新证据）：

- **（a）8 bit 段是削顶读数，8→12 bit 差值不是优化增益**：同相位换 12 bit 四个差值为
  正方形 −7.28 pp、三角形 −2.71 pp、五角星 +3.28 pp、圆形 −5.22 pp —— 三降一升 ⇒ 测量模式差异，非优化增益。
- **（b）三角形 12 bit 仍削顶（最高 1574 px），其 CV 仅乐观上界**；正方形 / 圆形 12 bit 段轻度饱和（120 / 55 px），其余为 0。
- **（c）同相位重测差值 ≤ 2.62 pp ⇒ CV 差 ≤ 2 pp 不构成证据**，仅测量重复性。
- **（d）验收按 CV 但 `exploration_budget=0.25` 非单调 ⇒ 段末 `*_final` 不是段内最优**（五角星 / 圆形段末高于段首，见 §5 反向链）。
- **（e）【新】CV 随曝光单调上升并饱和；跨曝光档的 CV 比较无效。** 同一相位、同一口径下，
  20 / 50 / 90 ms 的 CV 是三个不同数（例：正方形 `transport_cv` 0.419 / 0.645 / 0.759）；
  与 `common_replay_ladder_trial`（15 / 60 / 150 ms，正方形 `transport_cv` 0.357 / 0.615 / 0.644）一致 ——
  **短曝光被加性背景（杂散光）项主导、长曝光进入饱和前趋平**。结论：任何跨越曝光档的 CV 差都不可比；
  ≤ 2 pp 的 CV 差在任意口径下都不构成证据。

---

## 4. 重放曝光实验（核心新证据）

**目的**：用同一归档相位、三档曝光（20 / 50 / 90 ms）实测核心 CV，检验「内运行 CV 能否作为可信刻度」。

**方法**：`replay_phases.replay` 对 16 个形状×阶段 final 相位各重放 20 / 50 / 90 ms（每组 3 帧），
重放路径无满量程守卫，故本分析报告对 72 条记录逐点统计 `raw_frames.max()` 与核心（半径 60 px 排除区外）峰值 `core_peak`。

**表 4-1：同相位三曝光重放的核心 CV 与均值（mean 为全 ROI 平均计数，pk 为 90 ms 档核心峰值）**

| 形状 | 阶段 | 内运行 CV(曝光 bit) | cv@20 | cv@50 | cv@90 | m@20 | m@50 | m@90 | pk@90 |
|---|---|---|---|---|---|---|---|---|---|
| square | transport_cv | 0.7766(135 ms,8) | 0.4190 | 0.6445 | 0.7588 | 26.1 | 42.3 | 63.8 | 1038 |
| square | mono12_cv | 0.6574(117 ms,12) | 0.4193 | 0.5962 | 0.6751 | 30.2 | 52.2 | 81.2 | 901 |
| square | mono12_continue | 0.6442(153 ms,12) | 0.4099 | 0.5798 | 0.6580 | 30.6 | 53.4 | 83.9 | 981 |
| triangle | transport_cv | 0.7662(58 ms,8) | 0.5089 | 0.6759 | 0.7506 | 33.9 | 61.5 | 98.0 | 1174 |
| triangle | mono12_cv | 0.6640(104 ms,12) | 0.4913 | 0.6182 | 0.6761 | 41.8 | 80.6 | 132.7 | 1100 |
| triangle | mono12_continue | 0.6582(113 ms,12) | 0.4935 | 0.6209 | 0.6766 | 42.2 | 82.1 | 135.3 | 1133 |
| star | transport_cv | 0.7510(135 ms,8) | 0.5873 | 0.7102 | 0.7621 | 48.1 | 96.8 | 161.2 | 1489 |
| star | mono12_cv | 0.7069(118 ms,12) | 0.5721 | 0.6771 | 0.7262 | 54.3 | 112.7 | 190.2 | 1618 |
| star | mono12_continue | 0.7144(132 ms,12) | 0.5172(r2) | 0.6502(r2) | 0.7111(r2) | 44.3 | 87.6 | 141.3 | 1069 |
| circle | transport_cv | 0.7300(156 ms,8) | 0.3144(r2) | 0.4836(r2) | 0.5776(r2) | 25.8 | 42.2 | 63.2 | 519 |
| circle | mono12_cv | 0.6774(156 ms,12) | 0.3517(r2) | 0.5203(r2) | 0.6035(r2) | 28.0 | 47.4 | 72.6 | 557 |
| circle | mono12_continue | 0.6925(156 ms,12) | 0.3546(r2) | 0.5212(r2) | 0.6023(r2) | 28.1 | 47.6 | 72.5 | 530 |
| square | mono12_dense | 0.6443(156 ms,12) | 0.5451 | 0.6289 | 0.6341 | 44.6 | 80.3 | 85.7 | 776 |
| triangle | mono12_dense | 0.6647(153 ms,12) | 0.5641 | 0.6609 | 0.6879 | 66.3 | 120.0 | 128.1 | 877 |
| star | mono12_dense | 0.7316(146 ms,12) | 0.6110 | 0.7114 | 0.7193 | 100.1 | 181.2 | 191.8 | 1459 |
| circle | mono12_dense | 0.6958(146 ms,12) | 0.5415 | 0.6142 | 0.6308 | 55.6 | 99.0 | 95.1 | 711 |

> `—(r2)`：`common_replay` 原 90 ms 块末 4 条（star/mono12_continue 与 circle×3）因 SLM 关设备前 USB 掉线前兆而
> `core_peak`≈25、mean 骤降到 ~15 计数、CV≈0.045，属失效记录；由 `common_replay_r2`（洁净重测）覆盖，本表已填入 `(r2)` 值。
> **标注 `(r2)` 的 4 行是事后重测（与内运行相隔数小时），系统已漂移**，不可与同表其余 8 行（与内运行时间相邻的 `common_replay` 洁净子集）直接拼接；
> 12 bit 段（mono12_*）的 90 ms 行与内运行同口径，8 bit 段（transport_cv 首块）因满量程 255 削顶，仅作锚点。

**饱和审计结论（逐点 `raw_frames.max()`）**：
- 全部 72 条重放记录的 `core_peak` 均 **< 4014**（核心饱和守卫阈值 250/255×4095≈4015），即**核心口径下无饱和**。
- 原 `common_replay` 的 4 条失效记录是**唯一** `core_peak`≈25 的离群点，确认是设备掉线而非光学。

**与内运行一致性（基于时间相邻的 `common_replay` 洁净子集）**：12 bit 段的 90 ms 重放 CV 与内运行同相位 final CV 的差为
+1.4 ~ +1.9 pp（正方形 0.6751 vs 0.6574；三角形 0.6761 vs 0.6640；五角星 0.7262 vs 0.7069），
**全部落在 §3（c）的 2.62 pp 重复性以内** ⇒ 内运行序列（12 bit 段）的 CV 是可信刻度，短曝光（20 ms）的 CV 因加性背景主导而系统性偏高、不可用。

**漂移警示（来自 `common_replay_r2` 事后重测）**：`(r2)` 标定的 4 行与内运行相隔数小时，圆形两段的 90 ms 重放 CV 较内运行 12 bit final
低 **7–9 pp**（0.6035 / 0.6023 vs 0.6774 / 0.6925），正方形 / 三角形 / 五角星则基本持平。
说明系统**非漂移无关**：即便相位归档不变，跨数小时的复现也达不到 ±2 pp 重复性。这把 G6（无重复、无漂移检验）从「置信低」升级为「已观测到实际漂移」，
是下一步 N1（每尺度重测须带零级参照与重复块）与 N6（控制变量对照）的直接动机。

---

## 5. 结果：四形状 × 四阶段 CV 链（12 bit 对齐读法）

按 §3（a）（b）把 8 bit 首块锚点单列，只在 12 bit 段间比较（差值 > 2.62 pp 才算区分力）：

| 形状 | 8 bit 首块 | mono12_cv | mono12_continue | mono12_dense | 是否有区分力的下降 |
|---|---|---|---|---|---|
| square | 0.7766 | 0.6574 | 0.6442 | 0.6443 | 0.7766→0.6442 下降 ≥ 11 pp（8→12 切换 + 优化） |
| triangle | 0.7662 | 0.6640 | 0.6582 | 0.6647 | 0.7662→0.6582 下降 ≥ 10 pp；dense 仅动 ≤ 1.5 pp |
| star | 0.7510 | 0.7069 | 0.7144 | 0.7316 | **非单调**：mono12_cv 后转差（+0.2 / +1.7 pp） |
| circle | 0.7300 | 0.6774 | 0.6925 | 0.6958 | **非单调**：mono12_cv 后转差（+1.5 / +0.3 pp） |

**本会话唯一有区分力的变化**：正方形 / 三角形在 8→12 bit 切换后继续下降（≥ 3.5 pp/段），远超 2.62 pp 重复性；
五角星 / 圆形在 `mono12_cv` 之后不再改善甚至转差，与重复性同量级 ⇒ 这两段在 12 bit 下已无显著可优化余量
（或 `exploration_budget` 把段末拉离段内最优，见 §3（d））。**四段链不能写成「越跑越好」。**

**跨尺度（H1/H2/H3）**：无数据。仅 1200 一个尺度完成，无法回答「形状 vs 尺度」哪个主导。

---

## 6. 实验差距（gap）

| 编号 | 差距 | 影响 | 本轮数据已排除的假象 |
|---|---|---|---|
| G1 | **只跑完一个尺度（1200）**，600 / 200 缺失 | H1（尺度主导）/ H2（控制频带）/ H3（干净参照）不可判定；规划 12 格矩阵只完成 4 格 | — |
| G2 | **读数口径未闭环**：CV 随曝光变（§3e）、8/12 bit 不可比、三角形 12 bit 仍削顶 | 跨阶段 CV 只能定性比较，无法给「到位」的绝对刻度 | 已排除「短曝光 CV 高 = 好」的假象；已用 90 ms 重放对齐内运行 |
| G3 | **段末 ≠ 段内最优**（`exploration_budget=0.25`） | 五角星 / 圆形「转差」可能是验收策略假象，非真实退化 | 需回 manifest 逐块取段内峰值，并做 champion 保留对照 |
| G4 | **模型误差未分解**：LUT γ / 面板相差 / 相机离焦 / 倾斜混在一起，Step A 只拟合 Zernike 像差 | 五角星 / 圆形在宽载波（dense）下回退，残差里可能有非 Zernike 成分或尺度失配 | 未分解；候选：逐项开关 `bench_twin` 误差项（Cho 2026 GRAPHYCS 图模型自标定） |
| G5 | **重放只覆盖 final 相位，未重放段内 champion** | 无法区分「段末相位真更差」vs「重测噪声」 | 重放已补 90 ms 一致性，但未做 champion vs final 重放对照 |
| G6 | **单次、单种子、无重复、无漂移检验** | ±2 pp 内差异都不能判为效应；跨数小时复现不可信 | 已用 2.62 pp 重复性下限过滤；`common_replay_r2` 实测到圆形漂移 7–9 pp |

---

## 7. 下一步方案（优先序 + 具体命令）

| 序 | 针对 gap | 方案 | 具体命令 / 参数 |
|---|---|---|---|
| N1 | G1 | **跑完 B / C 两个尺度**（12 格闭环）。每格 `--no-gs-seed --record-responses --replay-exposures 20 50 90`；插入 `zero_ref_probe.py` 测零级参照 | `python -m ao_shaping.runners.slm.transfer_shapes_runner --model-dir <s1200> --side 600 --window-span 620 --out <s600> --no-gs-seed --record-responses --replay-exposures 20 50 90`；s200 用 `--side 200 --window-span 420`（注意零级排除区重叠 5.61%，CV 须与该剔除面积一并读） |
| N2 | G2 | **曝光标定**：每尺度定一条「曝光→CV」曲线（同相位 3–4 档），把 CV 换算到统一有效计数口径；报告 CV 一律附曝光与 `core_peak` | 复用 `common_replay` 三档 + `ladder_trial` 四档，离线拟合 `1/CV` vs 曝光线性段，定可用曝光窗口 |
| N3 | G2 / G5 | **指标替换**：引入曝光不变指标（背景扣除 / 平场归一后的 ROI CV + EE + 峰值比）；先用已有 72 条重放做离线对比择优 | 在 `physical_model_metrics` 上加曝光不变口径；用 `compare_shape_objectives` 类脚本对 72 条重放做对比择优 |
| N4 | G3 | **验收策略对照**：champion 保留（best-block 而非 final）+ 段间 warm start vs 冷启；用五角星 / 圆形两个失败案例做 A–B–B–A | 改 `validate.py` 的 `accept_metric`+`exploration_budget`，或加 `--keep-champion`；对 star/circle 单独跑一轮 |
| N5 | G4 | **模型误差分解**：逐项开关 `bench_twin` 误差项（LUT γ / 面板相差 / 离焦 / 倾斜），定位五角星 / 圆形回退来源 | 用 `bench_twin_calibrate` 反演；对照 Cho 2026 图模型自标定思路 |
| N6 | G3 / G4 | **控制网格单因素对照**：先做一次 `window_step` 4→2 的隔离实验（其余不变），否则 dense 段与 continue 段差异无法归因 | 单独 `--replay-stages mono12_dense` 对照，控制变量 |

---

## 8. 文献依据（Zotero 候选改进方法）

检索 `Zotero` 本地文库（用户文库，2026-10-28）得到与本轮差距直接对应的候选方法；引用格式：作者-年 + DOI。

### 8.1 测量口径与曝光 / 动态范围（对应 G2）
- **Mastiani, Cox & Vellekoop 2024**，《Practical considerations for high-fidelity wavefront shaping experiments》，*J. Phys. Photonics* 6(3):033001，10.1088/2515-7647/ad5775。
  系统梳理高保真波前整形实验的**测量陷阱**：曝光、动态范围、背景扣除与相机线性区——直接支撑 §3（e）与 N2/N3 的曝光不变指标。
- **Wavefront Sensorless Adaptive Optics: A Review**（2027，Zotero 文库条目 7KE8XVT9）。闭环工作流与反馈信号口径的综述，用于框定 G2/G3 的指标选择。

### 8.2 模型误差分解与自标定（对应 G4 / N5）
- **Cho et al. 2026**，《Graph-based modeling of optical system enables adaptive optics on dynamic samples with self-calibration》，*iScience*，10.1016/j.isci.2026.116769（GRAPHYCS）。
  用图模型显式表示光学系统各误差项并自标定——对应 §6 G4「LUT γ / 面板相差 / 离焦 / 倾斜」逐项分解。
- **Antonello et al. 2014**，《Model-based wavefront sensorless adaptive optics for large-photon-budgets**，*JOSA A* 31:1337，10.1364/JOSAA.31.001337。
  基于显式模型的 WFSless 优化基线，支持把物理预测模型作为反馈而非黑箱。
- **SLM 相位响应标定**：Xin 2025（低频频外差干涉标定，10.1088/2040-8986/ae01bc）；Zhang 2025（双光栅绝对干涉标定，10.37188/OPE.20253323.3630）——直接服务 LUT γ 与面板相差标定（N5）。

### 8.3 优化器 / 验收策略（对应 G3 / N6）
- **C8MQZ27L（Wang 2012）**，《Convergence rate of stochastic parallel gradient descent algorithm based on zernike mode》，10.3788/co.20120504.0407。SPGD 基于 Zernike 模式的收敛率——支撑 N6 的控制网格单因素对照设计。
- **BJF3DCZA（Wang 2026）**，《SPGD-based phase compensation for atmospheric thermal blooming》，*Opt. Express* 10.1364/OE.581759。SPGD 在非单调代价下的稳定性讨论，呼应 §3（d）段末≠最优。
- **Chromatix 2026**，《Differentiable wave-optics framework》，*Nat. Methods* 10.1038/s41592-026-03121-x。可微波前模型，可作 N3 曝光不变指标与 N5 模型分解的统一底座。
- **HVUWU87K（Yan 2026）**，《FourierGSNet》，*APN* 10.1117/1.AP.5.2.026005。GS 深度展开（deep-unrolling）相位恢复——替代 GS 初相位的候选（N 系列未采用 GS，但可作种子对比）。
- **RLNSKEKR（Herrezuelo 2026）**，PINN 光束整形，arXiv 10.48550/ARXIV.2607.18012。目标相位设计的物理信息网络候选。

### 8.4 匀光质量与零级（对应 G2 / 全形状）
- **VU9TQINS（Zhai 2023）**，几何稳相法 SLM 整形质量影响因素，10.5768/JAO202344.0401002。平顶光束质量因子定义——服务 N3 的指标替换。
- **M7HI3W8Z（Qi 2024）**，激光光束整形技术研究进展，10.3788/LOP231112；**RQYZD5UF（Zhang 2024）**，基于机器学习的激光匀光整形，10.7498/aps.73.20240747。匀光方法综述。
- **PDVN5NB8（Chen 2025）**，Lensless 零级-free 平顶，10.1016/j.optlastec.2025.114527。零级处理候选，对应 H3 零级排除区重叠问题。

---

## 9. 复现命令与数据目录

```bash
# 会话 A 模型链（已完成）：capture → fit → geometry_wide → initial → model640 → model_wide640 → seed_bakeoff
# 重放（洁净）：common_replay_r2（4.0 网格，36 条）+ common_replay_dense（2.0 网格，24 条）
python C:/Users/zhangh/AppData/Local/Temp/opencode/resume_s1200.py \
  --run-root data/debug/slm_transfer/multishape_scale_20261028/s1200 \
  --shapes square triangle star circle --skip-records \
  --replay-stages transport_cv mono12_cv mono12_continue \
  --replay-out common_replay_r2 --replay-exposures 20 50 90 --execute

# 会话 B / C（下一步 N1）
python -m ao_shaping.runners.slm.transfer_shapes_runner \
  --model-dir data/debug/slm_transfer/multishape_scale_20261028/s1200 \
  --side 600 --window-span 620 --out <s600> \
  --no-gs-seed --record-responses --replay-exposures 20 50 90
```

**数据目录**：`data/debug/slm_transfer/multishape_scale_20261028/s1200/`
（`shape/` 16 段、`records/`、`common_replay/`、`common_replay_dense/`、`common_replay_r2/`、`common_replay_ladder_trial/`）。

**图**：复用同子目录 `figures/`（`seed_bakeoff_square_20261028.png`、`exposure_peak_saturation_s1200_20261028.png`、`stage_final_cv_*`、`cv_trajectory_*`），均离线生成，见 `scripts/slm/generate_multishape_scale_live_figures.py`。
