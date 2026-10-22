# GSNet 实验系统实测指导（issue #165）

> 这一篇是**实验系统实测**的操作手册：一步步跑完「表征 → 测耦合 → 反演孪生 →
> 导出相位 → 迁移学习 → 闭环 A/B」，每一步给**可直接复制的命令**、**判据**、
> **回退方式**和**要记什么**。
>
> 结论先放前面（仿真里量出来的，见
> [`report/gsnet_aberration_transfer/report.md`](../../report/gsnet_aberration_transfer/report.md)）：
>
> * **默认的迁移学习配置会把网络弄坏**（6 个情形里 5 个变差，最差一格掉到平场
>   以下）。要用网络初值，必须同时做三件事：**算子换成这台设备拟合出来的孪生**
>   **＋ 只留整形项（相位项权重 0）＋ 验证集对比择优门**。
> * **可冻结的参数**：要质量用 `--freeze freeze-decoder`（可训练 336,097）；
>   要少学参数用 `--freeze freeze-encoder`（可训练 **86,065 = 模型的 21%**，
>   平均增益仍为正）。**不要用全参数微调**（平均增益 −0.0002，净负）。
> * **门 1 的阈值已经改好**（`--init-from-gsnet-trust-region-c-l2`，默认 2.5 rad）。
>   改之前门 1 把所有真实整形相位都挡下了，所以 **P1 验收条款从来没有真正跑过**；
>   改之后第一次真跑，结果是**门 3（验证集 Pearson）拒绝**、闭环安全回退。

---

## 0. 这一轮要回答的问题

> **在真实光路上，一个「初值」能不能帮闭环跳出局部最优？**

这和「网络相位本身好不好」是两个问题。**报告里必须按这个口径写**，不能把闭环
的最终分数说成网络的贡献。

一次会话要产出三样东西：

| 产物 | 回答什么 |
|---|---|
| 这台设备的**灰度→振幅耦合**（`c`, `ψ`） | 孪生里那条最要紧的物理，仓库里**从来没测过** |
| 这台设备的**拟合孪生**（JSON） | 让网络的前向是**这台设备**的模型，而不是理想模型 |
| **闭环 A/B** 的分数与判决 | 网络初值到底有没有用；没用就必须能安全回退 |

---

## 1. 前置条件（不满足就别开机）

- [ ] `docs/slm/pre_run_characterization.md` 的三个探针都跑过并且**可信**：
      `slm_drift_probe`（平场稳不稳、曝光线性吗）→ `slm_floor_probe`
      （噪声是读噪声还是漂移）→ `slm_abba_probe`（稠密随机相位测得出来吗）。
      **不要跳过** —— 探针不可信时，后面的每个数字都不可信。
- [ ] 曝光档位已经用 `slm_drift_probe` bracket 出来，并且**显式**传给后面的命令
      （默认路径不传曝光时它自己会探测，但探测值未必是你要的档位）。
- [ ] SLM 是 memory mode；**只允许 memory mode**（DVI 模式会挂死）。
- [ ] 面板不在冻结状态（`slm_diagnose` 的冻结检查过）。
- [ ] 探针相位用 `display_data(gray)` 写，**不传** `memory_number`
      （固件对当前槽是 no-op，面板不会刷新）**也不传** `wait_time_s`
      （翻转时间估算会报 0 ms）。采集一律走「稳定后再取」，不要裸 `sleep`。
- [ ] 预训练 checkpoint 在手：`data/gsnet_pretrain/20261022_coupling/checkpoint.pt`
      （在带耦合的孪生上重训的，2000 样本 / 24 轮）。

---

## 2. 阶段 0：测灰度→振幅耦合（**新增，迁移学习之前必须做**）

### 2.1 为什么

LCOS 的**实际振幅依赖驱动灰度**：平场相位在 1064 nm 下，0 阶桶能量随灰度呈
**~993 灰度周期**的变化（仓库的诊断判据就是它：「0 阶桶能量须有 ~993 灰度周期」）。
孪生现在按 `I(g) = 1 + c·sin(2π·grey/levels + ψ)` 建模这一项，但 **`c` 至今没有
被测过** —— 仓库里只有判据（`rel_spread > 0.15` ⇒ `c ≳ 0.075`），没有曲线。

不测就实验，网络会去利用一个**假定**出来的耦合。

### 2.2 测

```bash
# 稠密扫描 0..993（正好一个 2π 周期），Daheng 相机，0 阶桶能量
python scripts/slm/validate_flat_phase_gray.py \
    --cam-type daheng --cam-id 0 --slm-number 1 --wavelength 1064 \
    --exposure-ms <阶段0 定下的档位> --wait-time-s 0.3 --discard-count 3 \
    --scan --gray-step 25 --csv data/slm_gray_bench.csv
```

> ⚠️ 仓库里另一个 0 阶桶扫描（`slm-diagnose` 的调制检查）**只采 12 个点**
> （100 灰度步长），**会把周期里的峰与谷漏掉** ⇒ 它的 `rel_spread` 只是**下界**，
> 不足以定量拟合 `c`。要定量就得用上面这条稠密扫描。

### 2.3 拟合并写进孪生

```bash
python scripts/slm/fit_slm_amplitude_coupling.py \
    --csv data/slm_gray_bench.csv --metric bucket --levels 993 \
    --out data/debug/gsnet_hw_session/coupling_twin.json
```

它会打印：拟合出的 `offset / amplitude / period / phase`、换算到孪生参数化后的
**`c` 与 `ψ`**、以及实测的 `rel_spread`，并与判据 `rel_spread > 0.15` 对照。

> `--metric bucket` 用的是**冻结 0 级 ROI 的桶能量**（平场相位的 0 阶就是瞳面场的
> 直流分量，所以它才正比于实际振幅）。`sum` 是全帧总能量（含杂散光）、
> `max` 是峰值，两者只作旁证。CSV 里没有桶列时脚本会**退回并明确说一声**，
> 不会静默换列。

### 2.4 记什么

- `c`、`ψ`、拟合周期（应当 ≈ 993；偏离太多要查是不是灰度轴搞错了）
- 实测 `rel_spread` 与判据 0.15 的关系
- **拟合残差**（实测曲线 vs 模型曲线）—— 残差大说明单谐波模型不够用，
  此时别把 `c` 当真值用

---

## 3. 阶段 1：反演这台设备的孪生

### 3.1 采一批探针对

用 `slm-model-in-loop --debug` 采一批（**相位 → 帧**）对；它会落盘
`probes/*.npz`（每个含 `phase` 与 `model_grid`）与 `source_amplitude.npy`。

```bash
python src/ao_shaping/main.py slm-model-in-loop \
    --cam-type daheng --cam-id 0 --slm-number 1 --slm-wavelength 1064 \
    --region 64 --far-field-padding 8 --target-side 100 --cam-size 300 \
    --n-rounds 3 --probe-count 8 --holdout-probe-count 2 \
    --step-a-iterations 40 --step-b-iterations 200 \
    --exposure-time-ms <档位> --debug \
    --out data/debug/slm_model_in_loop/<ts>
```

### 3.2 拟合孪生

```bash
python scripts/slm/fit_bench_twin.py \
    --from-dir data/debug/slm_model_in_loop/<ts> \
    --fit-modes 4,5,6,7,11 --fit-lens-modes 4,5 \
    --levels 993 \
    --coupling-config data/debug/gsnet_hw_session/coupling_twin.json \
    --out data/debug/gsnet_hw_session/twin_fit.json
```

要点：

* **要反演哪些模式由你声明**（`--fit-modes` / `--fit-lens-modes`）—— 不写就不拟合。
* **耦合不是从探针拟合的**：model-in-loop 的探针命令的相位都差不多，
  耦合在探针上**不可辨识**。所以耦合用阶段 0 测出来的值，用 `--coupling-config`
  合并进去。
* **焦面标度必须用实测口径**。孪生的远场标度只通过焦距进入，所以「用哪个焦距」
  就是「用哪种标度口径」。上游实测这台实验系统的焦面标度是 **2.614022**
  （相机像元/模型采样点，相位斜坡扫描）⇒ **等效焦距 311.32 mm**；按标称焦距
  122.41 mm 算出来的解析标度只有 1.027778，**差 2.54 倍**。
  拟合脚本会**自动**从 `<debug 目录>/preflight/geometry.json` 读实测标度
  （也可以用 `--camera-px-per-model-px` 显式给）。读不到时会退回标称焦距并
  **明确警告** —— 那种孪生与实验系统的标度对不上，拟合出来的系数不可信。
* 面板 Noll 4（离焦）与相机离焦在远场里**近简并**，两者一起拟时要盯残差
  （脚本会提示这一点）。

### 3.3 记什么

- 每个被拟合参数的值与单位（LUT γ、离焦 m、每个 Zernike 系数 rad、倾斜 rad、
  耦合 `c` 与 `ψ`）
- **用的焦面标度口径**（实测值是多少、从哪来的、换算出的等效焦距）
- 拟合损失、两个阶段的损失、评估次数
- **拟合前后的残差对照** —— 拟合没改善初始配置时，要**照实说没拟合成功**，
  不要把失败的拟合当成结果

---

## 4. 阶段 2：导出网络相位（用拟合出来的前向）

```bash
python scripts/slm/export_gsnet_phase.py \
    --checkpoint data/gsnet_pretrain/20261022_coupling/checkpoint.pt \
    --source <source_amplitude.npy> --target <目标远场强度 .npy> \
    --twin-config data/debug/gsnet_hw_session/twin_fit.json --forward calibrated \
    --out data/debug/gsnet_hw_session/phase_calibrated.npy
```

⚠️ **换前向本身有代价**：checkpoint 内部是用**理想**前向训练出来的，直接把前向
换成拟合前向而不重训，迁移前的分数会掉（仿真里 panel-coma 从 0.3216 掉到
0.2636）。所以正确的顺序是 **反演 → 用拟合前向重训/重导出 → 再迁移**。

---

## 5. 阶段 3：迁移学习（微调）

```bash
python scripts/slm/gsnet_transfer.py \
    --checkpoint data/gsnet_pretrain/20261022_coupling/checkpoint.pt \
    --from-dir data/debug/slm_model_in_loop/<ts> \
    --twin-config data/debug/gsnet_hw_session/twin_fit.json \
    --freeze freeze-decoder \
    --gate \
    --epochs 20 --lr 1e-4 --batch-size 8 \
    --out data/debug/gsnet_transfer/<ts>
```

| 开关 | 为什么 |
|---|---|
| `--twin-config` | 整形项必须用**产生数据的那台孪生**去算，否则目标物理上达不到，梯度是噪声 |
| `--freeze freeze-decoder` | 要质量用这个（可训练 336,097，跨情形平均 Δ **+0.0191**）。要少学参数用 `freeze-encoder`（**86,065 = 21%**，Δ +0.0040）。**不要用 `full`**（Δ −0.0002，净负） |
| `--gate` | **必须有**。没有门，有害的微调总会被采纳；有了门，最坏情况从 −0.0837 限到 −0.0325 |

**看两个数**：

| 指标 | 含义 | 判据 |
|---|---|---|
| `phase_loss` | 相位项（对命令过的相位） | 随机猜测水平 = π²/3 ≈ **3.290**。低于它才算学到 |
| `shaping_loss` | 整形项（用标定前向把预测相位推到相机那张帧上） | 越低越好；没有绝对门槛 |
| `gate` | `"off"` / `"accepted"` / `"kept_pretrained"` | **`kept_pretrained` 意味着微调被判为有害、已丢弃**，产物就是未微调的那套 |

> 阶段 2 已经把相位项去掉了（仿真里**只留整形项**才有用；`w_phase=0`）。
> `gsnet_transfer.py` 目前相位项权重固定为 1 —— **若要复现仿真里那条最优组合，
> 需要确认它的相位项权重**；不确定就**先跑一次带门、冻结 decoder 的**，看门判什么。

---

## 6. 阶段 4：闭环 A/B（这一步才是「有没有用」）

```bash
# 对照臂：不用网络初值
python src/ao_shaping/main.py slm-model-in-loop \
    --cam-type daheng --cam-id 0 --slm-number 1 --slm-wavelength 1064 \
    --region 64 --far-field-padding 8 --target-side 100 --cam-size 300 \
    --n-rounds 3 --probe-count 4 --holdout-probe-count 2 \
    --step-a-iterations 40 --step-b-iterations 200 \
    --exposure-time-ms <档位> \
    --out data/debug/slm_model_in_loop/<ts>_physical

# 网络初值臂
python src/ao_shaping/main.py slm-model-in-loop \
    ... 同上 ... \
    --init-from-gsnet data/debug/gsnet_transfer/<ts>/phase.npy \
    --out data/debug/slm_model_in_loop/<ts>_gsnet
```

**两臂必须交错跑**（物理漂移是系统误差，交错才能消掉一阶项），每臂至少 3 次。

### 6.1 三道门与判决

| 门 | 卡什么 | 阈值 | 判决字符串 |
|---|---|---|---|
| 门 1 | GSNet 瞳面相位的 **rms（弧度）** | `--init-from-gsnet-trust-region-c-l2`，默认 **2.5** | `trust_region_rejected` |
| 门 2 | **实测**必须优于平场 | 严格大于 | `flat_bake_off_rejected` |
| 门 3 | 验证集探针的 Pearson | `--min-holdout-correlation` | `holdout_correlation_rejected` |

⚠️ **门 1 只是预筛，不是质量门。** 均匀随机相位的 rms 是 √(π²/3) = **1.81 rad**，
和真实整形相位（1.68 rad）几乎一样 —— **幅度本身分不出好相位和随机相位**。
真正表态的是门 2 和门 3。**报告里不要把门 1 写成质量判据。**

> 门 1 的阈值 2.5 rad 是从**实测尺度**来的：闭环自己的最优相位 rms 1.678 rad、
> 网络的整形相位 1.8–2.1 rad ⇒ 2.5 验证集余量，同时仍挡掉明显越界的相位。
> 原来的 0.5 rad 是给 Step A 的**系数步**定的，两个量不在同一个空间。

### 6.2 判据（issue #165 的 P1 条款）

> **网络初值运行不得劣于物理初值运行；劣化即关闭开关回退。**

* 网络臂的最终分数 **≥** 物理臂 ⇒ 通过。
* 网络臂更差 ⇒ 开关必须**自动关掉并回退**（门 2/门 3 就会这么做），
  这时**不算失败**，算「这条路上没有收益」。
* 报告里要写清楚**回退有没有生效**（判决字符串），以及**回退的代价**：
  多测一帧会扰动噪声流，所以回退臂的最终分与对照臂会有 ~1e-4 量级的差。

### 6.3 仿真里已经知道的

在**仿真**上用同一套门跑过（`data/debug/gsnet_init_sim_ab/gate_fixed/`）：

| 臂 | 判决 | 最终分 | 相对对照臂 |
|---|---|---|---|
| physical（对照） | `disabled` | 0.6765 | 0 |
| gsnet_full | **`holdout_correlation_rejected`** | 0.6768 | +0.0003 |
| gsnet_scaled | `holdout_correlation_rejected` | 0.6768 | +0.0003 |

**门 1 改好之前**，`gsnet_full` 是被 `trust_region_rejected` 挡下的（根本没测），
所以 **P1 条款从未真正被执行**。改好之后第一次真跑：**门 3 拒绝、闭环安全回退**、
最终分与对照臂持平。**这就是期望的「劣化即回退」行为。**

> 注意：这个仿真的「GSNet 臂」用的是**闭环自己的最优相位**当替身（不是真网络的
> 输出），所以它检验的是**门这套机制**，不是网络本身。实验系统上要用真相位。

---

## 7. 记录清单（写进报告用）

每次实验系统会话都要留下：

- [ ] 阶段 0 三个探针的结论（漂移 / 噪声类型 / 稠密相位可分辨性）
- [ ] 曝光档位与理由
- [ ] 耦合：`c`、`ψ`、拟合周期、实测 `rel_spread`、拟合残差
- [ ] 孪生拟合：每个参数 + 单位、两阶段损失、评估次数、拟合前后残差
- [ ] 迁移学习：`phase_loss`（对 π²/3）、`shaping_loss`、`gate` 判决、
      可训练/冻结参数数
- [ ] 闭环 A/B：两臂各次的分数、判决字符串、相对差、回退是否生效
- [ ] **每一条数字都要能追到命令**（命令 + 输出文件路径）

---

## 8. 已知会咬人的地方

1. **SLM memory slot**：连续写相位必须换槽；对当前显示中的槽写是 no-op，面板不刷新。
2. **曝光默认值**：不传曝光时代码会自己探测，但探测值未必是你要的档位 ——
   **显式传**。
3. **0 阶 ROI 要定位一次然后冻住**：逐灰度重新定位会跟着光斑走，把耦合曲线伪造出来。
4. **平场灰度 ≠ 相位 0**：SLM 有**振幅耦合**，平场灰度不同亮度不同（~993 灰度周期）——
   这既是本篇要测的东西，也是**所有以「平场」为基线的指标**的偏差来源。
5. **读出噪声会让 PIB > 1**：算框内能量占比前必须去本底（`prepare_frame`）。
6. **相位 wrap 会在 2π 边界产生高频衍射误差**：面板相位尽量少跳变。
7. **漂移**：平场峰值**不可复现**（实测），判漂移用**区域范数**而不是峰值。
8. **第一次开 SLM 后的前几帧不可信**：面板要弛豫，等稳定后再取。
9. **门 1 不是质量门**（见 §6.1）。
10. **焦面标度口径**：标称焦距与实测标度在这台实验系统上**差 2.54 倍**。拟合孪生必须用实测口径（见 §3.2）。
11. **`--init-from-gsnet` 的文件**：是**瞳面网格**的原始弧度（`region × region`），
    不是远场帧。

---

## 9. 相关文件

| 文件 | 作用 |
|---|---|
| `docs/slm/gsnet_transfer_hw_runbook.md` | 迁移学习的参数与调参指南（含冻结策略的完整对照表） |
| `docs/slm/gsnet_init_hw.md` | GSNet 初值闭环的实验系统手册（门 1 的来历与 P0 结果） |
| `docs/slm/pre_run_characterization.md` | **开机表征**（三个探针 + 危险默认值清单） |
| `report/gsnet_aberration_transfer/report.md` | 迁移学习 × 冻结策略 × 光路相差的**仿真**对照 |
| `report/gsnet_2f_calibration/report.md` | 前向校准与 P0 判据的**仿真**结果 |
| `scripts/slm/fit_slm_amplitude_coupling.py` | 灰度→振幅耦合：曲线 → `c`, `ψ` |
| `scripts/slm/fit_bench_twin.py` | 探针对 → 拟合孪生 JSON |
| `scripts/slm/export_gsnet_phase.py` | checkpoint + 目标 → 可下发的相位 |
| `scripts/slm/gsnet_transfer.py` | 迁移学习（`--freeze` / `--twin-config` / `--gate`） |
| `scripts/slm/gsnet_init_sim_ab.py` | 闭环 A/B 的仿真版（用来先验一遍门的行为） |
