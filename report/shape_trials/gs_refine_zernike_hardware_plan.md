# GS → Zernike → SPGD 匀化实验：实验测试计划

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`scripts/slm/run_gs_refine_zernike_sweep.py`](../../scripts/slm/run_gs_refine_zernike_sweep.py)
> **数据/关联脚本**: [`scripts/slm/generate_gs_refine_zernike_report.py`](../../scripts/slm/generate_gs_refine_zernike_report.py)
> **运行环境**: 硬件
> **说明**: GS → Zernike → SPGD 匀化实验的实验测试计划 (待执行, 非结论)；含现状 (仿真已确立的硬件无关结论 / 仿真天花板 / 合并上游 #140 的基线变化 / 复现性边界)、算法三阶段与锁定量、12 个原子化测试项 (H0~H12, 各带二元判据与止损)、执行顺序依赖图、交错重放等通用规程、数据记录规范与风险表。采集脚本见关联脚本
<!-- provenance:end -->

> **状态**：待实验执行（计划文档，非实验结论）
> **关联脚本**：[`scripts/slm/run_gs_refine_zernike_sweep.py`](../../scripts/slm/run_gs_refine_zernike_sweep.py)（采集）、[`scripts/slm/generate_gs_refine_zernike_report.py`](../../scripts/slm/generate_gs_refine_zernike_report.py)（离线出图）
> **仿真基线**：[`report/shape_trials/gs_refine_zernike_uniformity.md`](gs_refine_zernike_uniformity.md)
> **实验系统**：Santec SLM #1 `optionboardID 23020027` + 大恒 CCD `FJB24112232`，1064 nm，2f-Fourier 臂

---

## 1. 现状

### 1.1 已在仿真中确立、**与硬件无关**的结论（可直接采信）

| 结论 | 数值 | 依据 |
|---|---|---|
| Stage-B 拟合残差**随 `n_max` 单调不增** | 8 阶扫描 0.0719 → 0.0385 rad（−46.5%） | 嵌套最小二乘基的性质；`E3_orders` |
| 351 模（`n_max=25`）不触发 `ZERNIKE_CLIP=5.0` | `max|c|` 各阶一致 | `E3_orders` |
| `spgd_improved` 永远是**真实布尔决策**，不是缺字段 | — | `test_gs_refine_zernike_sweep.py::TestSpgdImprovedIsAlwaysRecorded` |
| `--lr 0` 复现已记载的**步长拒绝失效** | 5/5 阶 `spgd_improved=false` | `E6_lr0` |
| 能量守卫在仿真里**不是瓶颈** | 关掉守卫结果不变 | `E4_noguard` |
| 拟合函数**不原地修改入参**，各阶互相独立 | — | 直接验证 `fit_best_phase` |

### 1.2 仿真的天花板（本计划存在的原因）

数字孪生是纯 FFT，而 `_derive_target_side` 用的是**实验系统**模型（`camera_pixel_um=2.2`、焦距来自实测
tilt-shift scale），二者角尺度不匹配。于是 GS 预矫正在仿真里**打不过平场**（日志：
`GS warm start ... did not beat flat ...; keeping flat`），送进 Stage B 的 `best_phase` ≈ 平场：

| 量 | 仿真实测 | 含义 |
|---|---|---|
| `max|c|` | 各阶恒为 0.0142 rad | 拟合几乎没抓到任何结构 |
| `fit_rms / max|c|` | 2.7 ~ 5.1 倍 | 被拟合的相位接近平场 |
| `gs_freeform` vs `flat` PIB | 0.3450 vs 0.3448 | GS 相对平场**没有改善** |

**结论：仿真只能验证管线正确性，不能回答"多少阶 Zernike 才能复现 GS 整形"。**
阶数之间也无法在仿真里排序——PIB 跨阶极差 0.0046、CV 极差 0.0039，与**单轮 SPGD 抖动
0.0010~0.0035 同量级**，任何"最优阶数"都是在读噪声。这也是现有报告故意不给阶数排名的原因。

### 1.3 实验为什么能解锁

实验系统上 GS 相位是**真实且高度不均匀**的（`gs_uniformity_live_20261015` 实测剔零级
**CV ≈ 0.723**，面板梯度 RMS ≈ 0.052 rad/px）。这与仿真的"≈平场"是两个完全不同的
regime——`fit_rms` 会有真实结构可压缩，阶数效应会真正显现。**这是本计划必须实验的全部理由。**

### 1.4 合并上游后必须知道的变化

上游 issue #140（commit `540d51b2`）把 GS 最优相位从"获胜的 ±delta 探针"改成
"**干净的 dof 收敛点**"：

```python
-  best_phase = measured_phase   # 带 ±delta rad 随机扰动
+  best_phase = phase_dof        # 未扰动的收敛点
```

后果：拟合残差整体变好（`n_max=4`：0.2349 → 0.0826 rad），`max|c|` 从 0.05 降到 0.017。
**本计划所有基线数字都是 #140 之后的**，不要与 2026-10-16 之前的报告直接比较。

### 1.5 复现性边界（必须遵守）

实测：**完全相同的命令行**在 12 路并发与低负载下**逐位一致**（9 位小数全同）；
但 `--n-max` 列表或 `--max-energy-loss` 不同时，Stage A 的 `best_phase` 会整体改变
（同一 `n_max` 跨 run 的 `fit_rms` 相差 0.008~0.012 rad），而这两个参数都在 Stage A
**之后**才被读取——根因未查明。

> **执行纪律**：`fit_rms` **只在同一次 run 内跨阶可比**。任何跨 run 的残差绝对值比较一律无效。

---

## 2. 算法

### 2.1 三阶段数据流

```
Stage A  slm_gs_refine（GS 预矫正 + SPGD 细化）
         → best_phase（1200×1920 面板，弧度，raw 未包裹）
                │
Stage B  fit_best_phase(best_phase, n_max, beam_radius_px)
         → 瞳面裁剪 (pupil_side 900) → Noll 最小二乘拟合
         → coefficients[n_modes]（弧度）+ fit_rms_rad + max|c|
                │
Stage C  slm_zernike_pib（SPGD 在 Zernike 系数上）
         → Adam/AdaMOD 更新系数 → 冻结遮罩上重新打分
         → before/after {cv, pib, ee, ar, quality}
```

### 2.2 关键锁定量（改动任一必须整轮重扫）

| 量 | 值 | 为什么锁 |
|---|---|---|
| `beam_radius_px` | `450.0` | **同时**锁定 GS 瞳面与 Zernike 拟合基。换光路必须与 `--n-max` 一起重扫 |
| `slm.zernike_radius` | `450` | 驱动侧必须与 `--beam-radius-px` 一致，否则下发相位与拟合基错配 |
| 目标方框 | waist 推导（`resolve_target_side`） | 冻结遮罩，全程不得按 `argmax` 重定位 |
| 零级遮罩 | 平场定位，固定 | 整形帧最亮散斑会漂移，不能用整形帧 `argmax` |

### 2.3 度量口径

- 主指标：**剔零级 CV**（越低越匀），与 `gs_uniformity_live_20261015` 同口径以保证可比。
- 必须同时看 **EE**：单独优化 `-cv` 会掏空目标方斑（硬件实测 EE → 0.002）。
- 拟合侧指标：**相对 GS 全频随机相位的 far-field 复现保真度**（见 H6），这是本实验真正的科学量，
  `fit_rms` 只是相位空间的代理。

### 2.4 实验命令模板

```bash
# 采集（先跑基线，再跑扫描；--n-max 一次定好，中途不要改）
python scripts/slm/run_gs_refine_zernike_sweep.py \
  --cam-type daheng --cam-id 0 --exposure-time-ms 0.120 \
  --cam-size 256 --beam-radius-px 450 --slm-number 1 \
  --epochs 400 --gs-iters 200 --spgd-epochs 200 \
  --n-max 4 6 8 11 15 --lr 0.01 --delta 0.02 \
  --max-energy-loss 0.6 --seed 0 --save-images

# 离线出图
python scripts/slm/generate_gs_refine_zernike_report.py --run <输出目录>
```

---

## 3. 通用规程（每一项都适用，违反则数据不可用）

1. **交错重放**：每组候选按 `原相位 → 候选相位 → 原相位` 重放，第二轮逆序，取均值。
   这是本仓库既有纪律（`gs_uniformity_live_20261015` 即如此），用于抵消光路漂移。
2. **不追求逐位复现**：硬件闭环每次读出都带器件噪声。种子只锁定 SPGD 扰动符号，
   **不锁定测量**。判据用近似复现 + 重放配对，永远不用等号。
3. **固定 ROI 与遮罩**：整个 run 内不得重定位。
4. **先测噪声地板**：任何"改善"必须大于噪声地板才可采信（H2 是其余所有项的前置）。
5. **统计唯一入口**：配对检验 / `cohens_dz` / `holm_bonferroni` 一律用
   `ml/zernike/eval_stats.py`，**禁止在脚本里复制第二份实现**。
   配对轴只能是"同 seed 配 seed、同折配折"。
6. **背景处理**：CCD 帧含对称读出噪声，约一半像素为负。先减帧中位数再算 CV/PIB
   （`prepare_frame`），否则 `PIB > 1`。

---

## 4. 原子化测试项

每项独立可判定，有二元判据。`P0` = 阻塞其余全部。

### H0 · 实验系统基线与安全门 `P0`

- **目的**：确认光路可用、参数正确，且不会损坏设备。
- **前置**：SLM 已上电并物理复位（历史上有 DVI 模式挂起后必须断电的记录）。
- **步骤**：显示平相 → 定位光轴（平场 `argmax`，**不是**帧中心）→ 连续采集 20 帧。
- **判据**：曝光回读与设定一致；帧无饱和（`max_brt` 不顶格）；连续帧强度漂移 < 5%。
- **记录**：平场帧、曝光回读、光轴坐标、排除像素数。
- **风险**：SLM `open()` 静默挂起（曾观测 900 s 超时）。**必须用 memory 模式（`video_mode=0`），
  绝不自动尝试 DVI**。挂起后先确认无残留 python 进程，再给控制器物理断电。

### H1 · GS 预矫正 对比择优 `P0`

- **目的**：确认 GS 预矫正在**本实验系统**真的比平场好；否则整个 Stage B 无意义。
- **前置**：H0 通过。
- **步骤**：同一 seed 下跑 Stage A；交错重放比较 `flat` 与 `gs_freeform` 的剔零级 CV。
- **判据**：**GS 的 CV 显著低于平场**（差值 > 噪声地板的 3 倍）。若 GS 打不过平场，**立即停止**，
  先查 `--camera-pixel-um` / `--beam-radius-px` / 面板配准，不要继续烧 Stage C。
- **记录**：`best_stage`、`best_score`、`best_phase.npy`、两条基线的 CV/PIB/EE。
- **风险**：与 `gs_issue_live_20261015` 的失败模式一致（IoU 仅 0.068~0.098，未成实心方斑）。
  若复现该结果，本计划的 H3+ 需要先解决 GS 本身的质量问题。

### H2 · 噪声地板 `P0`

- **目的**：给出 CV / PIB / EE 的重复测量标准差，作为其余全部判据的分母。
- **前置**：H0 通过（**不需要** H1）。
- **步骤**：固定一个相位（平场 + 一个非平凡 GS 相位），各重复采集 30 帧，分组计算均值。
- **判据**：产出 `σ_CV`、`σ_PIB`、`σ_EE` 三个数及其置信区间。
- **记录**：原始帧、每帧曝光回读、三项 σ。
- **风险**：若 σ 与阶数间差异同量级（本仿真中正是如此），**H9 的阶数排序在物理上不可判**，
  必须如实报告而不是硬排。

### H3 · 真实 GS 相位的拟合单调性 `P1`

- **目的**：在**真实**（非平场）相位上验证 §1.1 的单调性是否依然成立。
- **前置**：H1 通过，拿到非平场 `best_phase.npy`。
- **步骤**：对**同一份** `best_phase` 在 `n_max = 3 4 6 8 11 15 20 25` 上拟合。
- **判据**：`fit_rms` 随 `n_max` **单调不增**（容差 1e-12）。出现上升即为 bug，不是发现。
- **记录**：每阶 `fit_rms`、`max|c|`、`n_modes`、是否 clip。
- **风险**：低。纯离线计算。

### H4 · 阶数拐点（knee）定位 `P1`

- **目的**：回答本实验的核心问题——**多少阶 Zernike 才够复现 GS 整形**。
- **前置**：H3 通过。
- **步骤**：用 H3 的数据找边际收益拐点（每新增 1 模带来的 `fit_rms` 下降跌破噪声地板处）。
- **判据**：给出一个**可复算**的拐点 `n_max*`，并报告其前后每模增益。
- **记录**：完整 `n_modes → fit_rms` 表 + 拐点判据。
- **风险**：拐点是**相位空间**的结论，未必等于 far-field 结论 → 必须配 H6。

### H5 · 系数幅值与基半径自洽 `P1`

- **目的**：确认拟合出的系数可安全下发，不会被 clip，且基半径没错配。
- **前置**：H3 通过。
- **步骤**：检查每阶 `max|c|`；用 `panel_phase_from_coefficients` 重建面板，与 `best_phase`
  在瞳面内逐点比较。
- **判据**：`max|c| < ZERNIKE_CLIP = 5.0` rad（留足裕量）；重建相位与原相位在瞳面内
  `max|Δ|` 与 `fit_rms` 自洽（同一残差量级）。
- **记录**：`max|c|`、重建残差、是否出现 clip 警告。
- **风险**：**基半径错配是本实验最隐蔽的失败模式**——它会让残差看起来"正常"但下发相位完全错。
  必须用重建比对兜住，不能只看 `fit_rms` 数值。

### H6 · far-field 复现保真度 `P1`（**核心科学项**）

- **目的**：Zernike 拟合**在远场**到底复现了多少 GS 整形效果。这是 `fit_rms` 的物理对应物。
- **前置**：H3、H5 通过。
- **步骤**：对每阶，把 `panel_phase_from_coefficients` 的结果下发（交错重放），
  在**冻结遮罩**上与 GS 全频随机相位、`flat` 三者比较 CV / PIB / EE / 综合分。
- **判据**：给出 `CV(n_max)` 曲线；判"该阶是否值得它的自由度"用
  **相对 GS 的 CV 差距是否 > 噪声地板**。
- **记录**：三方的 CV/PIB/EE/quality 表 + 重放原始帧。
- **风险**：这是**唯一**能把"阶数"和"匀化效果"联系起来的项。若曲线平坦，则结论是
  "Zernike 阶数不是瓶颈"，同样是有效结论，但必须这样写。

### H7 · Stage C 预算是否足够（补做被砍掉的 E1）`P2`

- **目的**：回答仿真里没答的问题——SPGD 不改善**到底是预算不足还是物理上限**。
- **前置**：H6 通过。
- **步骤**：`--spgd-epochs` 取 200 / 600 两档，其余完全相同，各自独立 run。
- **判据**：若 600 档的 `spgd_improved` 比例显著高于 200 档，则此前的不改善是**预算不足**；
  否则是**物理上限**。
- **记录**：每阶 `spgd_improved`、`J` 的 min/final、实际执行轮数（守卫跳过多少）。
- **风险**：算力。实验系统每轮 SPGD 是一次真实相机曝光，必须先用小预算估算再决定 600 档是否可行。

### H8 · Zernike 受限 vs 全频随机相位 `P2`

- **目的**：把"用 Zernike 近似"这件事本身的代价量化出来。
- **前置**：H6 通过。
- **步骤**：同一 seed 下比较 `gs_freeform`（全频随机相位，已被 SPGD 细化过）与最佳阶数的
  `spgd after`。
- **判据**：CV 差距 > 噪声地板 → Zernike 受限确有代价；否则可放心用低阶 Zernike 提速。
- **记录**：两者的 CV/PIB/EE 与差值。
- **风险**：两者若用了不同预算则不可比；必须锁定 `epochs`/`spgd_epochs`/`seed`。

### H9 · 带噪声地板的阶数排序 `P2`

- **目的**：在**有统计功效**的条件下给阶数排序，替代仿真的"不可排序"。
- **前置**：H2、H6 通过。
- **步骤**：每阶跑 6 个 seed（seed 是配对轴），用 `ml/zernike/eval_stats.py` 做
  sign-flip 精确置换检验 + `cohens_dz` + Holm-Bonferroni。
- **判据**：`min_attainable_pvalue(n=6) = 2/2⁶ = 0.03125`。**达不到就不给 p 值，只报效应量**；
  跨度与噪声地板同量级时**明确拒绝排序**（沿用现有报告 `scan_verdict` 的做法）。
- **记录**：逐 `(n_max, seed)` 原始值、检验统计量、校正后 p 值。
- **风险**：6 个 seed 已是 6 次完整 run 的实验系统时间。若预算不足，**降到 4 seed 并声明
  `min_attainable_pvalue = 0.25`**，只报效应量——不要用 4 折假装能判显著。

### H10 · 能量守卫消融（实验系统）`P3`

- **目的**：仿真里守卫无影响，实验系统读出噪声更大，需重测。
- **前置**：H7 通过。
- **步骤**：`--max-energy-loss 0.6` 与 `0` 两档，其余相同。
- **判据**：报告守卫跳过轮数占比，以及关闭后 CV 是否变化（> 噪声地板才算变）。
- **记录**：每档的 CV/PIB、跳过轮数。
- **风险**：关掉守卫可能让能量崩塌的相位被采纳。**若 CV 改善但 EE 明显下降，判为失败**
  （这正是"只优化 `-cv` 会掏空方斑"的已知反模式）。

### H11 · 步长 / 学习率 `P3`

- **目的**：确认 `--lr 0.01 --delta 0.02` 在真实系数尺度上是合适的起点。
- **前置**：H7 通过。
- **步骤**：`--lr` 取 0 / 0.01 / 0.03 三档。
- **判据**：`--lr 0` 应复现已知的步长拒绝失效（`spgd_improved=false`）；非零档应至少不劣于 0.01。
- **记录**：每档 CV/PIB、`J` 轨迹。
- **风险**：实验系统时间昂贵。若 H7 已显示实验系统读出噪声主导，先跳过本项。

### H12 · 基半径自洽扫描 `P3`

- **目的**：确认 `beam_radius_px=450` 是 GS 瞳面与 Zernike 基的**共同**正确半径。
- **前置**：H5 通过。
- **步骤**：`--beam-radius-px` 取 400 / 450 / 480，驱动侧 `zernike_radius` 同步改动。
- **判据**：450 若是真值，其远场复现保真度应优于两侧。
- **记录**：每档 `fit_rms` 与 far-field CV。
- **风险**：改半径必须**同时**改驱动侧，否则下发的是错配基——见 H5。

---

## 5. 执行顺序与依赖

```
H0 实验系统基线  (P0)
├── H1 GS 对比择优  (P0)
│   ├── H3 单调性
│   │   ├── H4 knee
│   │   └── H5 系数自洽
│   │        └── H6 far-field 保真 ★  ← 核心科学项
│   │             ├── H7 预算是否足够
│   │             │    ├── H10 守卫消融
│   │             │    └── H11 步长
│   │             └── H8 受限 vs 逐像素相位
│   └── H9 带噪声地板的排序
└── H2 噪声地板  (P0，可与 H1 并行准备)

H5 ──> H12 基半径扫描
```

- `H6` 是**核心科学项**，其余都是支撑或消融。
- `H2` 不依赖 `H1`，可与 `H1` 并行准备。
- 任何 `P0` 项失败 ⇒ **停止**，不要继续往下烧实验系统时间。

---

## 6. 数据记录规范

- 每次 run 一个时间戳目录，`summary.json` + `spgd_nmax<n>.csv` + `best_phase_nmax<n>.npy`，
  `--save-images` 打开以留远场图。
- `data/debug/` 是 gitignored，**只提交报告与图，不提交产物**。
- 报告必须经 `scripts/slm/generate_gs_refine_zernike_report.py` 生成；
  `report/README.md` 由 `scripts/sync_report_provenance.py` 重生成，**禁止手工编辑**。
- 每项测试的判据、原始数据路径、失败/成功结论都要写进对应 `report/shape_trials/*.md`。
- **噪声地板（σ_CV/σ_PIB/σ_EE）必须写进任何一篇引用本计划数据的报告**，
  否则读者无法判断某个差异是真实效应还是噪声。

---

## 7. 风险与止损

| 风险 | 触发信号 | 处置 |
|---|---|---|
| GS 打不过平场 | H1 判据失败 | 停止；查 `--camera-pixel-um` / `--beam-radius-px` / 面板配准 |
| SLM `open()` 挂起 | 日志止于"成功打开SLM" | 强杀 → 确认无残留进程 → **物理断电**；只用 memory 模式 |
| 相邻写入同一 memory slot | 面板不刷新 | 每次写随机 slot（2..125）且排除当前显示槽 |
| 遮罩被 `argmax` 拖走 | 指标跳变不连续 | 只在**平场**帧定位光轴一次并冻结 |
| 阶数差异淹没在噪声里 | H2 的 σ ≥ 阶数间差异 | 如实报告"不可排序"，不得硬给排名 |
| 只看 CV 变好、EE 崩塌 | H10 出现该模式 | 判为失败；目标函数必须含能量项 |
| 实验系统时间耗尽 | H9 跑不完 6 seed | 降到 4 seed 并声明 `min_attainable_pvalue = 0.25`，只报效应量 |

---

## 8. 明确**不**在本次实验范围内

- 仿真 GS 前向模型标定（已确认 OOM：GS bench FFT 网格会到 7200²–14400²）。
- 自由像素 GS / 可微分整形（Zernike 低阶基**物理上无法**合成方形远场，见仓库 ANTI-PATTERNS）。
- 复核上游 blaze（`--blaze-period-px`）路径；本计划默认 blaze 关闭（`blaze_period_px=-1`，
  仿真下开启会直接抛错）。
