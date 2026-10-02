# Zotero 扫描：整形目标函数 / 评价函数目录

扫描对象：本地 Zotero 库（快照见文末「数据来源与安全性」）。
目标：为 `src/ao_shaping/optimizer/wfless/slm_square_shaping.py` 的 `objective=` 提供可实现的候选度量。

## 阅读约定

- **来源标签**（严格区分，不含推测）
  - `PDF正文` — 从 PDF 文本层直接提取，公式逐字转录
  - `PDF正文(几何重建+文本层双向确认)` — 文本层线性化丢失分式结构，先由字形坐标+分数线几何重建，再由纯文本提取独立复核一致
  - `Zotero笔记(未校验)` — 仅来自 Zotero 笔记，**未经 PDF 校验**
  - `NOT RECOVERED` — 公式无法获得，禁止以任何形式补全
- **方向** — 该文明确写出的优化方向。凡论文未写方向者标 `未说明`。
- **权重** — 论文给出的数值权重。绝大多数单指标论文**未给权重**，一律记 `未给`。
- 约定：正文中 OCR/文本层把希腊字母与下标压平（如 `PGauss`、`SVL1`）。本文按上下文还原排版，但**不还原论文未写出的量**。

## 元数据校验

所有条目均经快照库 `itemData / itemCreators / itemAttachments` 交叉核对。**注意**：部分条目 Zotero 的 `creator` 字段为空，引用信息取自 PDF 首页；此类已在条目中注明。

---

## 1. 平顶/整形质量（直接对应本项目方形整形）

### 1.1 光束均匀性 δ 与均方根误差 I_RMSE

- **来源**：`PDF正文`
- **原文**：Zhai Zhongsheng 等，2023，*Analysis of factors affecting beam shaping quality of geometric steady-phase method based on SLM*，Journal of Applied Optics 44(4)
- **item key**：`VU9TQINS` ｜ DOI `10.5768/JAO202344.0401002`（库内已核对）
- **本地 PDF**：`Zotero/storage/B9KCQFZZ/Zhongsheng 等 - 2023 - Analysis of factors affecting beam shaping quality of geometric steady-phase method based on SLM.pdf`（副本 `MF7C87BH`）
- **公式**（式 17、18）：

  ```
  δ      = (1 − I_RMSE) × 100%                             (17)
  I_RMSE = sqrt( Σ_{(u,v)∈S} [ I(u,v) − Ī ]² / (n − 1) )   (18)
  Ī      = (1/n) Σ_{(u,v)∈S} I(u,v)      （正文随式给出）
  ```

- **符号**：`I(u,v)` 取样区 `S` 内像素光强；`S` 方形平顶光斑取样区；`n` = `S` 内取样点数；`Ī` = `S` 内平均光强。强度单位任意（a.u.），`δ` 为百分比。
- **方向**：`δ` **MAXIMIZE**（原文：「`IRMSE` 的值越小，输出光束的均匀性越高」）。
- **场景/归一化**：无强度归一化 —— `I_RMSE` 是**绝对**标准差，对整体光强涨落敏感。
- **权重**：`未给`。
- **原文实测**：稳相法 98.3%（仿真）/ 83.9%（实验，含零级光）；叠加闪耀光栅移除零级光后 97.2%。

### 1.2 能量利用率 η

- **来源**：`PDF正文` ｜ 同上 `VU9TQINS`
- **公式**（式 16）：

  ```
  η = ( ∬_v I_out ds ) / ( ∬_v I_in ds ) × 100%
  ```

- **符号**：`I_in` = SLM 充作反射镜（无调制）时后焦面测得输入光强；`I_out` = 后焦面测得输出光强；`v` = 积分区域。
- **方向**：**MAXIMIZE**（能量利用率）。
- **权重**：`未给`。
- **备注**：需一次「无调制」参考测量，故非纯单帧目标。
- **原文实测**：移除零级光后 72.3%（未移除 91.3%）。

### 1.3 顶部不均匀度 δ（相对量，无量纲）★最贴近仓库现有 `uniformity_cv`

- **来源**：`PDF正文`
- **原文**：张旭 等，2024，*基于机器学习的激光匀光整形方法*，物理学报
- **item key**：`RQYZD5UF` ｜ DOI `10.7498/aps.73.20240747`（库内已核对）
- **本地 PDF**：`Zotero/storage/XFMV97NE/张旭 等 - 2024 - 基于机器学习的激光匀光整形方法.pdf`
- **公式**（式 1、2）：

  ```
  δ = sqrt( Σ_{(x,y)∈W} [ ( |I(x,y)| − Ī ) / Ī ]² / (n − 1) )      (1)
  Ī = (1/n) Σ_{(x,y)∈W} |I(x,y)|                                    (2)
  ```

- **符号**：`I(x,y)` 衍射面光强分布（含均匀光斑区与周围非均匀区）；`W` 均匀光斑区域；`n` = `W` 内采样点数；`Ī` = `W` 内平均光强。
- **方向**：**MINIMIZE**（原文：「顶部不均匀度值越小…光强分布越均匀」）。
- **归一化**：**相对**（除以 `Ī`），故无量纲、对整体光强涨落不敏感 —— 与 §1.1 的 `I_RMSE` 关键差别。
- **权重**：`未给`。
- **与仓库对照**：`src/ao_shaping/drivers/sim/slm_shaping_bench.py:291` 的 `uniformity_cv` = `vals.std()/vals.mean()`，与本式**同族**（仅分母 `n` vs `n−1`、以及是否取 `|I|` 的约定差异）。
- **原文实测**：机器学习补偿使顶部不均匀度相对降低 13%。

### 1.4 多点聚焦均匀度（标准差）

- **来源**：`PDF正文`（散文定义，无公式）
- **原文**：刘卉 等，2023，*一种通用的反馈式波前整形优化算法改进策略*，Acta Photonica Sinica
- **item key**：`93LPW5XT` ｜ DOI `10.3788/gzxb20235206.0629002`（库内已核对）
- **本地 PDF**：`Zotero/storage/E2STEEPN/Liu Hui 等 - 2023 - 一种通用的反馈式波前整形优化算法改进策略.pdf`
- **定义**：多个焦点的**标准差**（原文：「均匀度定义为多个焦点的标准差」）。
- **方向**：`未说明`（作为 NSGA-II 双目标之一，NSGA-II 通常同时最小化各目标）。
- **公式**：**NOT RECOVERED** —— 该文只给散文定义，未转写公式；其引用 [23] Feng Qi 等 2019 Opt. Express 是原始出处。

### 1.5 峰背景比 PBR

- **来源**：`PDF正文`（散文定义，无公式）
- **原文**：`93LPW5XT`
- **定义**：二元振幅型调制下，评价函数为「聚焦光斑与散斑背景光强的平均值的比值」（Peak-to-background Ratio, PBR）。
- **方向**：**MAXIMIZE**（原文以 PBR 升高为收敛）。
- **公式**：**NOT RECOVERED**（散文定义）。
- **权重**：`未给`。

### 1.6 增强因子（透过散射介质单点聚焦）

- **来源**：`PDF正文`（散文定义，无公式）
- **原文**：`93LPW5XT`
- **定义**：优化后聚焦点的光强 与 优化前散斑平均光强 之比。
- **方向**：**MAXIMIZE**。
- **公式**：**NOT RECOVERED**（散文定义）。
- **权重**：`未给`。

---

## 2. 无波前传感 AO（SPGD 类）远场指标

出处统一为：许振兴 等，2021 博士论文《基于深度强化学习的自适应光学波前控制研究》第三章 §3.1「目标函数」。
**item key** `D9ADBVTS`；引用文献 [147]、[25]。库内 `date = 2021`，`publicationTitle` 与 `DOI` **均为空**。

> **读取状态说明**：该条目附件为 `attachments:Physic/adaptive optics/..._许振兴.caj`（CAJ 格式），**原始文件当前不存在**（实测 `FILE NOT FOUND`）；`storage/NUWTVR75/` 下仅存 Zotero 自身的全文缓存 `.zotero-ft-cache`（178,978 字节）。下列公式转录自该缓存（即 Zotero 对原 PDF 的抽取），公式编号与上下文完整，但属**间接**来源。标记为 `PDF正文(Zotero全文缓存)`。

### 2.1 斯特列尔比 SR

- **公式**（式 3-10）：

  ```
  J_SR = I(x, y) / I₀(x, y)
  ```

- **符号**：`I(x,y)` 畸变波前远场图像**峰值强度**；`I₀(x,y)` 理想平面波前远场图像**峰值强度**。
- **方向**：**MAXIMIZE**（原文：「`J_SR` 越大校正效果越好」）。
- **归一化**：峰值强度比；**无参考测量即可**（理想峰值可由光功率/孔径解析给出）。
- **权重**：`未给`。
- **原文评价**：「计算简单，但空间频谱低，无法直观反映整体光强分布。」「极易受大气湍流闪烁的影响，容易陷入局部极值，因此具有局限性。」

### 2.2 像清晰度 IS

- **公式**（式 3-11）：

  ```
  J_IS = Σ_{y=0}^{M−1} Σ_{x=0}^{N−1} I(x, y)
  ```

- **符号**：`I(x,y)` 像素强度；远场图像大小 `M × N`。
- **方向**：**MAXIMIZE**。
- **归一化**：**无** —— 未做强度归一化，故对整体光强起伏敏感。
- **权重**：`未给`。
- **原文评价**：「测量容易且能够直观反映光强分布，具有较高的空间频谱。」

### 2.3 归一化像清晰度 UIS

- **公式**（式 3-12）：

  ```
  J_UIS = ( Σ_{y=0}^{M−1} Σ_{x=0}^{N−1} I(x,y) ) / ( Σ_{y=0}^{M−1} Σ_{x=0}^{N−1} I₀(x,y) )
  ```

- **符号**：`I₀(x,y)` 理想（参考）图像像素强度。
- **方向**：**MAXIMIZE**。
- **归一化**：除以参考图总强度 —— 原文动机：「使像清晰度函数能够适应光强存在整体强度起伏的应用环境」。
- **权重**：`未给`。

### 2.4 环围能量 EE

- **公式**（式 3-13）：

  ```
  J_EE = Σ_{(x,y)∈S} I(x, y)
  ```

- **符号**：`S` 为**理想衍射的艾里斑区域**；原文注明「实际应用中为了便于计算，`S` 采用以艾里斑直径为长度的方形区域」。
- **方向**：**MAXIMIZE**（原文：「`J_EE` 越大校正效果越好」）。
- **归一化**：**无**（未除以总强度或理想值）。
- **权重**：`未给`。

### 2.5 平均半径 MR（光斑二阶矩 / 能量扩展度）★对仓库是新增量

- **公式**（式 3-14）：

  ```
  J_MR = ( Σ_{y=0}^{M−1} Σ_{x=0}^{N−1} [ (x − x_c)² + (y − y_c)² ] · I(x,y) )
         / ( Σ_{y=0}^{M−1} Σ_{x=0}^{N−1} I(x,y) )
  ```

- **符号**：`I(x,y)` 像素强度；`(x_c, y_c)` 图像**质心**坐标；远场图像 `M × N`。
- **方向**：**MINIMIZE**（原文：「`J_MR` 越小校正效果越好」）。
- **归一化**：除以总强度（原文：为适应整体强度起伏做了归一化）。
- **权重**：`未给`。
- **原文评价**：「最大优点是不仅能够反映整体光强分布且能够直观反映远场光斑的能量扩展度。」
- **原文收敛速度对比**（61 单元 DM，双边 SPGD，达到收敛值 80% 所需迭代）：`J_IS` 1354、`J_EE` 1275、**`J_MR` 361**（最少，但计算量大）。

### 2.6 SPGD 梯度估计式（辅助，非目标函数）

同一来源给出的更新律，便于对照本项目 `_spgd_sign` / 双边差分实现：

```
ΔJ_t      = J(U + ΔU, t) − J(U, t)                       (3-1)  单边
∇_U J(U,t) ≈ η₁ · ΔJ / ΔU                                (3-4)
ΔJ_t      = J(U + ΔU, t) − J(U − ΔU, t)                  (3-5)  双边
∇_U J(U,t) ≈ η₂ · ΔJ / ΔU                                (3-6)
U_t       = U_t + α ∇_U J(U, t)                           (3-7)
```

超参：`η₁`、`η₂` 超参数；`α` 步长因子（学习速率）。收敛缩放经验：`T_M ≈ M·T₁`（式 3-8）；带宽 `f_s ≤ f_c/(30·M)`（式 3-9，`f_c` = CCD 帧频）。

---

## 3. 远场光斑二阶矩 / 探测器加权（几何类）

### 3.1 远场光斑均方半径（波前本征模校正）★对仓库是新增量

- **来源**：`PDF正文`
- **原文**：梁佳新、向汝建、杜应磊、顾静良、吴晶，2020，*基于变形镜本征模式和远场测量的光束净化*，强激光与粒子束 32(8): 081002
- **item key**：`F76LQAY8` ｜ DOI `10.11884/HPLPB202032.200082`（**取自 PDF 首页**；库内 `creator`/`publicationTitle`/`DOI` **均为空**）
- **本地 PDF**：`Zotero/storage/PB5NPFM6/基于变形镜本征模式和远场测量的光束净化.pdf`（9 页）
- **公式**（式 6）：

  ```
  g = ⟨r²⟩ = ( Σ I(x,y)·[ (x − x̄)² + (y − ȳ)² ] ) / ( Σ I(x,y) ) = μ Σ_i a_i²
  ```

- **符号**：`I(x,y)` 远场光斑任一位置光强；`(x̄, ȳ)` 光斑中心（远场光斑半径）；`μ` 与光学系统有关的常数；`a_i` 第 `i` 项本征模式系数。
- **方向**：**MINIMIZE**（作为评价函数，`⟨r²⟩` 越小越好；原文由式 (17) 显式给出校正量 `a_i^−corr = −a_i`，即朝 `a_i → 0` 收敛）。
- **理论基础**（原文式 1）：`∫_A |∇φ|² dA ∝ ⟨r²⟩`（几何光学下波前相位梯度模平方与远场光斑均方半径成正比）。
- **权重**：`未给`。
- **正文位置**：p3「所以评价函数为 `g = ⟨r²⟩`」；p4 给出 `g₀ = μΣa_i²` 与模式系数的关系。
- **备注**：与 §2.5 `J_MR` 数学形式**等价**（皆为强度加权二阶矩），但本文给出 `μΣa_i²` 的模式基分解，使其可做**逐模式** 3 测量求解（式 17），无需直接算质心。

### 3.2 二次加权探测器信号 W（大像差 WFSless AO）★对仓库是新增量

- **来源**：`PDF正文`
- **原文**：Booth Martin J.，2007，*Wavefront sensorless adaptive optics for large aberrations*，Optics Letters 32(1)
- **item key**：`ANSGMX63` ｜ DOI `10.1364/OL.32.000005`（库内已核对）
- **本地 PDF**：`Zotero/storage/9TH6KTN7/Booth - 2007 - Wavefront sensorless adaptive optics for large aberrations.pdf`
- **公式**（式 6）：

  ```
  W = ∬ I(ξ, η) · D(ξ, η) dξ dη
  ```

  探测器灵敏度原文给出两种取法：
  - `D = ξ²`（无限探测器）—— 在几何光学区等价于「乘以总强度的均方光斑半径测量」，但零像差处给**极小**而非极大；
  - `D = 1 − ξ²/R²`（`ξ < R`），`ξ ≥ R` 时 `D = 0`（有限探测器）—— 「we obtain a representation of W that has the desired properties」（有唯一极大、球对称）。

- **符号**：`ξ, η` 探测器平面极坐标；`I` 焦面光斑强度分布；`R` 选取的探测器半径。径向坐标归一化使圆孔衍射极限焦斑**第一强度零点**位于 `ξ = 1.22λ`。
- **方向**：**MINIMIZE**（`W` 在零像差处为极大；原文以迭代压缩 `W` 实现校正）。
- **权重**：`未给`。
- **相关式**（原文式 5）：`σ² = (π NA / λ)² Σ b_{n,m}² = (π NA / λ)² ||b||²` —— Lukosz–Zernike 展开下均方光斑半径与系数向量的模平方成正比（`b_{n,m}` LZ 系数，`NA` 数值孔径）。
- **注意**：Booth 的极小化方向与 §3.1/§2.5 的「二阶矩越小越好」一致，但 `D = 1 − ξ²/R²` 的**权重形式**不同 —— 它是 `ξ²` 的径向截断加窗，等价于「对半径 `R` 内的二阶矩做能量归一化加权」，可抑制大尺度离轴能量主导。

### 3.3 掩膜探测器信号 MDS ★对仓库是新增量

- **来源**：`PDF正文`
- **原文**：Linhai Huang & Rao，2011，*Wavefront sensorless adaptive optics: a general model-based approach*，Optics Express 19(1)
- **item key**：`6N87VBR5` ｜ DOI `10.1364/OE.19.000371`（库内已核对）
- **本地 PDF**：`Zotero/storage/QLK4EY4K/Linhai和Rao - 2011 - Wavefront sensorless adaptive optics a general model-based approach.pdf`（副本 `FVI6HL97`）
- **公式**（式 7）：

  ```
  MDS(x', y') = ( ∬_{r < R} I(x', y') dx' dy' ) / ( ∬ I(x', y') dx' dy' )
  ```

- **符号**：`I(x',y')` 远场强度；`R` 探测器半径；分母为全帧积分（原文：「the whole expression of the right-hand side of Eq. (5) is also normalized by dividing by the sum of `I(x',y')`」）。
- **方向**：**MAXIMIZE**（`MDS` 大 = 中心小桶内能量占比高 = 像差小）。
- **配套关系**（式 8）：`SM ≈ c₀ (1 − MDS)` —— 波前梯度二阶矩 `SM` 与 `1 − MDS` 成正比，`c₀` 为由探测器半径 `R` 决定的趋势线斜率（实测例：`R = 16 D_L/λ`，`c₀ = −194.1`）。
- **权重**：`未给`。
- **Strehl（该文评价用）**（式 17）：`SR = P[I(x,y)] / P₀[I₀(x,y)]`，峰值功率比，**MAXIMIZE**。
- **备注**：`MDS` 与仓库 `power_in_bucket` 同族，差别在分桶形状（**圆形** `r<R` vs 目标方形支撑）。

### 3.4 单模光纤耦合效率 η ★最简单的可实现目标

- **来源**：`PDF正文`（§5 所引论文 §3.2）
- **公式**：

  ```
  η = P_prop / P_init
  ```

- **符号**：`P_prop` = 传播进光纤的光功率；`P_init` = 进入光纤前的光场功率。
- **方向**：**MAXIMIZE**。
- **权重**：`未给`。
- **原文物理依据**：「As higher-order modes, such as the L1 and L-1 modes, are orthogonal to the Gaussian mode, they can not be coupled to a single-mode fiber under normal circumstances. Any imperfections in the Gaussian mode lead to a loss of coupling efficiency and any imperfections in the higher-order modes lead to them having a higher than zero coupling efficiency.」
- **⚠ 已知失效模式**（作者实测）：像差化的 L1 模中心强度仍可为零，导致无耦合、适应度良好，但模本身仍明显带像差 ——「an aberrated L1 mode can still keep the intensity at its center at zero, leading to no coupling and a good fitness value, even though the mode is still noticeably aberrated.」
- **可实现性**：只需功率计，不需相机；对本项目无单模光纤的场景**不适用**，但作为「最简标量目标」的对照基线有价值。

---

## 4. 归一化桶内指标与复合 FOM

### 4.1 归一化桶内功率 nPIB

- **来源**：`PDF正文`
- **原文**：DiComo Gregory P. 等，2025，*Beaconless adaptive optics for atmospheric laser propagation with multi-plane convolutional neural network*，Optics Express 33(15)
- **item key**：`VHLEEXGR` ｜ DOI `10.1364/OE.561077`（库内已核对）
- **本地 PDF**：`Zotero/storage/G6B488KP/DiComo 等 - 2025 - Beaconless adaptive optics for atmospheric laser propagation with multi-plane convolutional neural n.pdf`（副本 `2QZD7WX2`）
- **公式**（式 6，文字定义）：

  ```
  nPIB = ( 系统落在桶内的功率占比 ) / ( 无湍流、同孔径、同直径理想高斯焦斑落在桶内的功率占比 )
  桶径：D_bucket = 2.5 · L·λ / D_telescope
  ```

- **符号**：`L` 望远镜到目标面距离；`λ` 波长；`D_telescope` 望远镜口径。
- **方向**：**MAXIMIZE**。原文：「当光束全部功率落入桶内时，`nPIB` 饱和于略大于 1 的值。」

### 4.2 桶内斯特列尔比 SIB

- **公式**（式 7）：

  ```
  SIB = max_B I(T, C) / max_B I(vac)
  ```

- **符号**：`T` 湍流；`C` 校正相位；`vac` 无湍流真空传播；`B` 桶（大小与位置同式 6）。
- **方向**：**MAXIMIZE**。

### 4.3 复合 FOM（几何平均）

- **公式**（式 8）：

  ```
  FOM ≡ sqrt( nPIB · SIB )
  ```

- **方向**：**MAXIMIZE**。
- **原文给出的性质**（三条，均为显式陈述）：
  1. 「`FOM = 0` when either `nPIB` or `SIB` is zero.」
  2. 「agnostic to the relative importance of each metric, in the sense that doubling either `nPIB` or `SIB` and halving the other leaves `FOM` unchanged.」
  3. 「when `nPIB ∼ SIB`, `FOM ∼ nPIB ∼ SIB`.」
- **权重**：`未给`（几何平均**隐含**等权，可视为对数域 `w = 0.5/0.5` 的算术平均 —— 但论文未以权重形式表述）。
- **增益因子**（式 15）：`g = FOM_ML / FOM_track`，相对仅跟踪(tracking-only)解的标量增益（「an AO system which achieves a gain of `g = 2` would be equivalent to doubling the system laser energy」）。

### 4.4 SSIM 复合评价指标（附**显式数值权重**）★对仓库是新增量

- **来源**：`PDF正文`
- **原文**：Wang Shuai 等，2026，*Study of SPGD-based phase compensation for atmospheric thermal blooming in the upward transmission of high-energy lasers*，Optics Express 34(7): 12452
- **item key**：`BJF3DCZA` ｜ DOI `10.1364/OE.581759`（库内已核对）
- **本地 PDF**：`Zotero/storage/KDJZLADJ/Wang 等 - 2026 - Study of SPGD-based phase compensation for atmospheric thermal blooming in the upward transmission o.pdf`（11 页）
- **公式**（式 5，p4）：

  ```
  J      = w₁ · J_I + w₂ · J_SSIM                          (5)
  J_SSIM = 1 − SSIM
  ```

- **符号与标量定义**：
  - `I` = **衍射极限光斑半径内**的平均强度（原文：「the average intensity, I, within the diffraction limited spot radius is used」）。
  - `J_I` = 当前光斑的 `I` 与衍射极限光斑的 `I` 之**差**（原文：「`JI` denotes the difference in `I` between the current spot and the diffraction limited spot」）。
  - `SSIM` = 结构相似度（引用 [35]），比较「当前扰动相位下的远场光斑」与「衍射极限光斑」。
  - `w₁`、`w₂` = 两项权重。
- **方向**：**MINIMIZE**（原文显式把 SSIM 转成不相似度以纳入极小化：「To convert SSIM into a minimization objective, it is transformed into a dissimilarity metric」；「A smaller `J_SSIM` indicates that the spot energy distribution is closer to the diffraction-limited spot」）。
- **权重**：**`w₁ = w₂ = 0.5`（有据）** —— 原文 p4：「It can be seen that the compensation is relatively better and more stable when `w₁ = w₂ = 0.5`」（`ND = 28` 条件下比较多组权重组合）。
- **设计动机（对本项目高度相关）**：「However, `I` is highly sensitive to the selected circular region. Using `I` alone as the evaluation metric can lead to fluctuations in compensation performance.」→ 即**单一圆形 ROI 内平均强度指标对 ROI 半径敏感，会让 SPGD 性能抖动**，故用 SSIM 复合项稳定。
- **配套性能指标**（非目标函数）：`I/I_{ND=1}`，其中 `I_{ND=1}` 为弱湍流（`ND = 1`）下的平均强度；原文报告 `ND < 1000` 时峰值强度可达衍射极限的 75% 以上。
- **更新律**（式 6、7，Adam 形式）：`∇J(a_{k+1}) = (J⁺−J⁻)·a_k`；`a_{k+1} = a_k − α·m̂_{k+1}/(√v̂_{k+1}+ϵ)`，`α` 学习率，`m̂`/`v̂` 梯度一二阶矩估计，`ϵ` 任意小正数。
- **扰动项**（式 4）：`∆P(x,y) = Σ_{i=1}^{N} a_i Z_i(x,y)`，`N = 30`（本文取 30 项 Zernike）。

---

## 5. 遗传算法适应度（LG 模态圆对称性）

- **来源**：`PDF正文`
- **原文**：Roivainen Mikko，2024，*Aberration correction of a spatial light modifier with a genetic algorithm*，**Tampere University 学士学位论文（Bachelor's thesis）**，Faculty of Engineering and Natural Sciences，Examiner: Associate Prof. Robert Fickler，2024 年 5 月
- **item key**：`T663WY5B` ｜ 30 页 ｜ 库内 `date`/`publicationTitle`/`DOI` **均为空**
- **本地 PDF**：`Zotero/storage/MT6VG7S5/Roivainen - Aberration correction of a spatial light modifier with a genetic algorithm.pdf`

> ⚠ 更正记录：本文早期草稿误记为「密歇根大学硕士论文」，经 PDF 首页核对**已更正为 Tampere University 学士论文**。

- **优化方向（原文显式）**：「The GA offered by MATLAB attempts to minimize any fitness function given to it, and therefore any fitness function should give a lower fitness value for a better result.」→ 全部适应度 **MINIMIZE**。
- **探针场**：`L1` / `L−1`（Laguerre-Gaussian 甜甜圈模）；每代每个体测 3 次（高斯 + L1 + L−1）。

### 5.1 模式功率比适应度

- **来源**：`PDF正文`（p19，式 3.1 逐字确认）
- **公式**：

  ```
  F = − P_Gauss / ( (P_L1 + P_L−1) / 2 )              (3.1)
  ```

- **符号**：`P_Gauss` 高斯光束测得功率；`P_L1`、`P_L−1` `L1`/`L−1` 模测得功率。
- **方向**：**MINIMIZE**（原文：「The sign of the fitness value is changed to account for the GA function trying to minimize it.」）。
- **备注**：相机方案下用「光束中心像素求和」近似各功率；被单模光纤方案采用（§3.4）。

### 5.2 模式功率 + 相关性适应度（**已被作者实测证伪**，留作反例）

- **来源**：`PDF正文`（p21，式 3.2 逐字确认）
- **公式**：

  ```
  F = ( − P_Gauss + P_L1 + P_L−1 ) · Corr              (3.2)
  ```

- **方向**：**MINIMIZE**。
- **原文证伪结论**：「This assumption was found to be not true as the GA converged onto modes with non-symmetric intensity distributions.」「Aberrated LG modes with opposite ℓ signs were meant to not correlate well, but the GA managed to find a set of coefficients that gives the opposite modes an identical circularly asymmetric intensity distribution.」→ **不建议实现**。

### 5.3 圆对称性适应度（最终版，成功收敛）

- **来源**：`PDF正文(几何重建+文本层双向确认)`
  - 首次提取时文本层把分子/分母线性化，顺序不可辨，按字形 y 坐标 + 两条分数线（`y=507.0`，`x=[271.4,328.5]` 与 `x=[330.9,402.3]`）重建。
  - **复核**：p22 纯文本提取给出 `F = −2PGauss/(PL1+PL−1) · 2/(SVL1+SVL−1)`，与几何重建**一致**。
- **公式**（式 3.3，两个相邻分式相乘）：

  ```
              2·P_Gauss        2
      F = − ─────────── · ─────────────                (3.3)
            P_L1 + P_L−1     SV_L1 + SV_L−1

      等价于  F = − 4·P_Gauss / [ (P_L1 + P_L−1)·(SV_L1 + SV_L−1) ]
  ```

- **符号**：
  - `P_Gauss`、`P_L1`、`P_L−1` 同 §5.1（像素求和近似）。
  - `SV_L1`、`SV_L−1` = 各半径上**角度方差之和**。原文三步法：① 沿以模式中心为心的圆周插值取强度，**每圈 100 个等角点**，自中心像素外 **10 px** 起每 **2 px** 增一径；② 对每圈 100 点求方差；③ 把所有圈的方差相加。
  - 完全对称时 `SV = 0` → `F → −∞`（强惩罚）。
- **方向**：**MINIMIZE**。
- **原文评价**：「This method performed successfully in testing, converging on a corrective aberration with good mode quality.」
- **局限（作者自述）**：「measuring the fitness value of a single individual takes approximately 3 seconds and each generation requires checking 200 individuals… the scale of improvement… will decrease until it is on the same scale as the measurement variance, at which point the best fitness value will stop improving and simply oscillate.」「the fitness value alone is not a good indicator of whether the method is functioning as it should」

---

## 6. 矢量自适应光学（偏振域指标）

- **来源**：`PDF正文`
- **原文**：Ma Yifei 等，2024，*Vectorial adaptive optics for advanced imaging systems*，Journal of Optics
- **item key**：`CJKTWIHF` ｜ DOI `10.1088/2040-8986/ad4374`（库内已核对）
- **本地 PDF**：`Zotero/storage/6Y9BFME7/Ma 等 - 2024 - Vectorial adaptive optics for advanced imaging systems.pdf`
- **方向**：两个指标均 **MINIMIZE**。

### 6.1 矢量校正精度 P

- **公式**（式 1）：

  ```
  P = c · ‖ S − S̃ ‖        c = 1/2
  ```

- **符号**：`S = [S₁, S₂, S₃]` 目标 Stokes 向量（归一化庞加莱球坐标，各分量 ∈ [−1,1]）；`S̃` 实际 Stokes 向量；`c` 归一化因子，`=1/2` 因 `‖S − S̃‖ ∈ [0,2]`。
- **适用域**：「valid over a line, a region of interest, or the whole pupil area」。

### 6.2 Stokes 均匀度 U

- **公式**（式 2、3）：

  ```
  U = c · sqrt( ‖ S̃ − S̄ ‖² )        c = 1
  S̄ᵢ = (1/n̄) Σ S̃ᵢ                    (3)
  ```

- **符号**：`S̃ = [S̃₁, S̃₂, S̃₃]` 实际 Stokes；`S̄ = [S̄₁, S̄₂, S̄₃]` 全瞳平均 Stokes；`c = 1`（「due to the symmetrical nature of Stokes parameter values」）。
- **物理含义**：原文定义为「实际 Stokes 相对平均 Stokes 的标准差」，度量重排序（re-sorting）光场的均匀性。
- **⚠ 文本层缺损**：式 (2) 的分母/归一化项在线性化后不可辨，**完整 `U` 公式标记为 NOT RECOVERED**（上文仅保留可确认的 `c·√(‖S̃−S̄‖²)` 骨架与 `c = 1`）。
- **适用域**：「valid over the whole imaging pupil or sub-regions … can be used to evaluate global uniformity」。
- **备注**：**偏振/矢量域**，与本项目标量强度整形不同轴；仅在需要矢量整形时适用。

---

## 7. 机器学习损失函数（训练用，非硬件目标函数）

### 7.1 均方误差 MSE

- **来源**：`PDF正文`
- **原文**：张旭 等，2024（`RQYZD5UF`）
- **公式**（式 4）：

  ```
  L = (1/m) Σ_{i=1}^{m} ( y_i − Y_i )²
  ```

- **符号**：`y_i` 网络输出；`Y_i` 真实标签；`m` 样本个数。
- **方向**：**MINIMIZE**。
- **⚠ 归类说明**：这是**神经网络训练损失**，不是作用在光学系统上的评价/目标函数。仅当用可微模型闭环优化 SLM 相位时才充当目标函数（参见本项目 `algorithm/signal_processing/differentiable_beam.py` —— ⚠️ 2026-10-01 修正：原文写的 `algorithm/differentiable_beam.py` 不存在，顶层 `algorithm/` 只剩 `__init__.py` 与 `base.py`）。**不可**直接作为 `slm_square_shaping.py` 的 `objective=` 硬件模式。

### 7.2 混合策略光子学（未校验笔记）

- **来源**：`Zotero笔记(未校验)`
- **原文**：Zhou Shiyun 等，2025，*Hybrid strategy in compact tailoring of multiple degrees-of-freedom toward high-dimensional photonics*，Light: Science & Applications，DOI `10.1038/s41377-025-01857-3`（库内已核对；`date = 2025-04-21`）
- **item key**：`ZVNJ7Y22`
- **附件**：`attachments:Physic/光场调控/Zhou 等 - 2025 - ....pdf`（链接式）。**实测 `Zotero/storage/` 下无对应 PDF**（`rglob("*Hybrid strategy*")` 返回空），故无法校验。
- **状态**：笔记提及 `L(E′, E)` 型 MSE 损失；公式 **NOT RECOVERED**，**不作为目录条目采纳**。

---

## 8. 未能提取公式的条目（诚实登记）

| item key | 条目 | 状态 |
|---|---|---|
| `725NVRNW` | 祁家琴 等，2024，*激光相干合成系统中基于自适应随机扰动电压的 SPGD 优化算法*，激光与光电子学进展 61(15)，**DOI `10.3788/LOP231638`**（取自 PDF 首页；库内 `DOI` 为空），CNKI 文件名 `JGDJ202415025` | **不是目标函数来源。** 主附件在 `D:\storage\...`（实测不存在）；副本 `Zotero/storage/RQ9ESP26/`（9 页）可读。内容为 Piecewise SPGD 的**扰动电压与增益分段策略**；用中英双语目标函数词表扫描，**含等号的公式段命中数 = 0** |
| `69GZ4QN9` | Zhao Zimo 等，2025，*Intensity adaptive optics*，Light: Science & Applications，DOI `10.1038/s41377-025-01779-0` | 传感器件为相机的**强度 AO** + 无感知相位 AO 串联；22 页版与 9 页版（`4FSW2VYA`/`6SCY7FDU`/`8SZ3UEYL`/`EJVQYHX7`/`IIZI3UF9`/`IUAIRS8A` 共 6 份副本）。扫描到的相关式为 Pancharatnam 相位连接式 (17) `Π(j₁,j₂) = arg(j₁^H j₂)` 与 (16)，**未找到标量 merit 公式** |
| `M7HI3W8Z` | 亓岩 等，2024，*激光光束整形技术研究进展*，Laser & Optoelectronics Progress，DOI `10.3788/LOP231112` | PDF 文本层**无任何含等号的公式段**（疑为图片/扫描件）。**NOT RECOVERED** |
| `4FNPL8CY` | Schmidt M. 等，2024，*Dynamic beam shaping—Improving laser materials processing via feature synchronous energy coupling*，CIRP Annals，DOI `10.1016/j.cirp.2024.05.005` | 全文无标量评价函数；讨论焊接热载荷/熔深/孔隙等工艺结果。**该文不提供可实现的标量 merit** |
| `RLNSKEKR` | Herrezuelo Rafael de la Fuente，2026，*Physics-informed neural networks for optimal beam shaping in flat optics*，arXiv `10.48550/ARXIV.2607.18012`（v2） | 为 PINN **物理损失**（无解析标量 merit）。检索命中的 "sharpness" 是**定性描述**，非公式。**不适用** |
| `D35EBAU2` | *deep learning based phase retrieval with complex beam shapes for beam shape correction*，DOI `10.1364/OE.547138` | 库内 `creator` 与 `date` **均为空**；附件为 `text/html`（ResearchGate 页面），无 PDF 正文可提取 |
| `D9ADBVTS` | 许振兴 2021 博士论文 | 原始 `.caj`/PDF 不存在；仅 `.zotero-ft-cache`。详见 §2 读取状态说明 |
| `6N87VBR5` | Linhai & Rao 2011 | 主附件为 `attachments:` 链接式（CAJ）；副本 `QLK4EY4K` 可读，公式见 §3.3 |

---

## 9. 更正记录（相对本文档早期草稿）

本节记录草稿中**经核对被证伪并已删除/更正**的内容，以免后续引用出错：

1. **删除** `725NVRNW`「归一化环围能量 `J = EE_farfield/EE_ideal`，值域 0..1」条目及基于它的实现建议 —— 该论文公式段命中数为 0，**不存在该公式**。
2. **删除** `69GZ4QN9` 的 `ΔP_n = P_n^target − P_n^*` —— 全文检索**不存在**该式（疑与 `BJF3DCZA` 式 (4) 的相位扰动 `∆P(x,y) = Σa_iZ_i(x,y)` 混淆）。
3. **更正** `F76LQAY8` 期刊：实为 **强激光与粒子束 / High Power Laser and Particle Beams** 32(8): 081002，DOI `10.11884/HPLPB202032.200082`，**非**「光学学报」。
4. **更正** `T663WY5B` 文献类型：实为 **Tampere University 学士学位论文**（2024-05），**非**「密歇根大学硕士论文」。
5. **降级** `CJKTWIHF` §6.2 的 `U` 公式为 **NOT RECOVERED**（分母在线性化文本层不可辨），仅保留可确认骨架。
6. **升级** `T663WY5B` 式 (3.3) 为 **双向确认**（几何重建 + 文本层复核一致）。

---

## 10. 与仓库现有实现的对照

现有实现：`src/ao_shaping/drivers/sim/slm_shaping_bench.py`
（`power_in_bucket` L263、`uniformity_cv` L291、`strehl` L309、`zero_order_fraction` L326、`compute_metrics` L350、`composite_from_pib_cv` L379）

| 本目录条目 | 仓库现状 | 结论 |
|---|---|---|
| §1.3 顶部不均匀度 δ（`RQYZD5UF`） | `uniformity_cv` = `std/mean` | **已覆盖**（同族；分母 `n` vs `n−1` 差异可忽略） |
| §2.4 环围能量 EE / §3.3 MDS | `power_in_bucket` | **已覆盖**（圆形桶 vs 方形支撑为可选变体） |
| §1.1 `I_RMSE`（`VU9TQINS`） | 无 | **新增可选**：绝对均匀性，对整体强度涨落敏感的对照项 |
| §2.5 `J_MR` / §3.1 `⟨r²⟩` | 无 | **新增推荐**：强度加权二阶矩，MINIMIZE，文献收敛迭代数最少（361 vs 1354/1275） |
| §4.4 `J = w₁J_I + w₂(1−SSIM)` | 无 | **新增推荐**：唯一带**显式数值权重**（`w₁=w₂=0.5`）的复合指标；专门解决「单一 ROI 平均强度对半径敏感 → SPGD 抖动」 |
| §3.2 Booth `W`（`D = 1 − ξ²/R²`） | 无 | **新增推荐**：对离轴大能量鲁棒的加窗二阶矩 |
| §4.3 `FOM = √(nPIB·SIB)` | 无 | **新增推荐**：单一标量同时约束桶内能量与峰值；任一为 0 则 FOM 为 0 |
| §2.1 `J_SR` | `strehl()` **名不副实** | ⚠ 见下 |
| §6.1/§6.2 `P`/`U` | 无 | 矢量/偏振域，与标量强度整形不同轴 |

### ⚠ 命名冲突（需决策）

`slm_shaping_bench.py:309` 的 `strehl()` 实现为**均值中心化后的余弦相似度**（`np.dot` of mean-subtracted, normalised arrays），**不是**经典 Strehl 比。本目录 §2.1（`J_SR = I/I₀` 峰值强度比）与 §4.2（`SIB = max_B I/max_B I_vac`）才是经典定义。三者物理含义不同（形状匹配 vs 峰值比），**不应互换**。

建议：新增模式时避免复用 `strehl` 一词，例如 `peak_ratio`（§2.1）、`strehl_in_bucket`（§4.2），并为现有 `strehl()` 更名或加注释。

### 与仓库既有反模式的对应

- §4.4 的设计动机（ROI 半径敏感 → 性能抖动）与 `AGENTS.md` 中「Re-locating the target ROI by `argmax` on **every** iteration」「Computing `PIB`/`CV` on a **raw** CCD frame」两条反模式**同源**。若实现 §4.4，ROI 必须**冻结**且帧须先做中值去噪/截零（参见 `slm_gs_refine._prepare_frame`）。
- §2.5 `J_MR` 与 §3.1 `⟨r²⟩` 依赖质心 `(x_c,y_c)`。`AGENTS.md` 已警示 `reset_window()` 返回中心不可信、0-order 须 `argmax` 重定位 —— 实现时须遵守同一规则。

---

## 11. 建议的实现优先级

按「对方形平顶整形的可实现性 × 文献支持强度 × 是否带可引用权重」排序：

1. **`mean_radius`**（§2.5 / §3.1）— MINIMIZE。实现最简（只需 `intensity` 与质心），文献给出明确迭代数优势（361 vs 1354/1275），且是仓库当前缺口。
2. **`ssim_composite`**（§4.4）— MINIMIZE，`w₁=w₂=0.5` 直接取自原文。**本目录中唯一带显式数值权重的候选**，且直击 SPGD 单标量抖动问题。
3. **`fom_geomean`**（§4.3）— MAXIMIZE，`√(nPIB·SIB)`；需理想参考（可由无调制平场一次测量得到，与仓库 `slm_gs_refine` 的 bake-off 思路一致）。
4. **`booth_window`**（§3.2）— MINIMIZE，`D = 1 − ξ²/R²` 加窗二阶矩；对零级光/离轴散斑鲁棒（`VU9TQINS` 报告零级光是均匀性主要劣化源）。
5. **`irmse`**（§1.1）— MINIMIZE，`sqrt(Σ[I−Ī]²/(n−1))`；与现有 CV 构成「绝对 vs 相对」对照。
6. **`peak_ratio`**（§2.1）— MAXIMIZE，`I/I₀`；需注意原文自述「极易受大气湍流闪烁影响」，在硬件上应先做中值去噪。

新增任一模式都应遵循仓库既有约定：在 `objective_mode` 分派而非新增 runner；ROI 中心与硬件 `argmax` 规则一致且**冻结**；指标在去噪帧上计算。

---

## 数据来源与安全性

- 原始库 `C:\Users\zhangh\Zotero\zotero.sqlite` 在扫描时有活跃 WAL 与 7 个 Zotero 进程，**全程未写入**。
- 快照：`C:\Users\zhangh\AppData\Local\Temp\opencode\zlib\zotero.sqlite`（连同 `-wal`、`-shm` 一并复制后恢复）。
  - `PRAGMA quick_check` → `ok`
  - 条目 11,703；笔记 2,836；`itemDataValues` 27,532
- PDF 读取：`PyMuPDF 1.27.2.3`，全部**只读**打开 Zotero `storage/` 下的副本；路径一律用 `Path.glob()` 解析（实测存在手敲路径与真实文件名差 1 字符导致 `FILE NOT FOUND` 的坑）。
- 全文检索：262 条笔记命中目标函数/评价指标类关键词（中文 + 英文，大小写不敏感）。
- 元数据校验脚本：`C:\Users\zhangh\AppData\Local\Temp\opencode\verify_meta.py`（条目）、`verify_pdfs.py` / `verify_round3.py` / `verify_round4.py` / `verify_round5.py`（正文与公式复核）。
- 备用：`C:\Users\zhangh\Zotero\zotero.sqlite.bak`（未使用）。