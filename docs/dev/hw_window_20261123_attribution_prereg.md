# 下轮实验窗口预注册：方向门 fail 归因（映射闭合 → 方向门重判）· 2026-11-23

> **状态**：预注册草案，待用户批准后方可开窗；判据在采集前冻结，采后不回改。
> **上游状态入口**：[hw_plan_consolidated_20261118.md](hw_plan_consolidated_20261118.md)（§3 决策门 G1–G4）。
> **本窗口要回答的唯一问题**：11-20 方向门 pooled 4/14 的失败，是**模型算子错**还是**几何/映射错**，
> 以及**测量本身有没有分辨力**。不做效率标定、不做 LUT 发布、不改 #247 映射标定。

---

## 0. 为什么开这个窗口（背景与已排除项）

### 0.1 映射异常已被零设备定位（2026-11-23 原始帧重定位结论）

对 11-23（60 帧）/ 11-21（44 帧）/ 11-21b（60 帧）原始帧重定位（扣 0 级 PSF、ROI 由预测位置导出、
150 px 全峰表）得到：

- **全幅 1D 图样（逐行完全相同，即 fy≡0：ramp_x / grating_x / tilt_x）落在 CCD 列轴**，
  而其位移幅值 = K·N（K=6.344 px/条纹、K×1200=7613、34 瓣 rms 1.06 px）本身与 2f 解析（6.2975）
  和 A 映射（6.4177）一致 ⇒ **错的只是轴向，不是标度**。
- 全幅 1D **y** 族（fy≠0）落列轴正确；全幅 2D `blaze2d` 与局部 patch/ladder 映射正常
  （`blaze2d` 独立复核残差 ≤1.8 px）。
- **11-23 的 `scale_gate.y` fail（0.664）是假象**：`grating_y_p256` 真 ±1 在 **29.4–30.4 px**，
  恰落在 `core_exclude=34.96 px` 盲区内；11-23 记的 88.89 px 是 **±3 级**（3×29.42≈88.3；
  ±2 因 50% 占空比严格为零；±3/±1 强度比 0.094–0.102 ≈ 理论 1/9）。
- ⇒ **A 映射（2D / 局部内容类）可冻结**；**全幅 1D fy≡0 类不得用 A 映射校验**。
- 根因只能落在 SDK/固件 `SLM_Disp_Data`（`libs/SLM_DLL_ver.2.51`，仓库外）或面板 10-bit 内部存储路径
  —— 仓库软件链（生成 L258-530、上传 L1434/L1502-1503、驱动 L1507-1524）已逐段验证 1:1 无转置无重采样
  ⇒ **零设备不可定级**，需设备侧最小判别实验（S1b）。

### 0.2 已排除 / 尚未排除

**已排除（有实测证据）**：

- 不是输入相位错：14 帧 `panel_gray` 逐像素 mismatch = 0；用 `panel_gray` 反解相位替换前向输入后
  方向 cosine 仍 ≈0（`model_direction_hw_20261117`）。
- 不是索引错位：7×7 模型×实测交叉匹配对角 6/7 占优，无 sina/cos 互换（`agent_notes_20261118.md`）。
- 不是单纯噪声主导：逐模 SNR 2.26–5.04，噪声相关上限 0.914–0.981 ≫ 实测相关（同上）。

**尚未排除（本窗口的主靶）**：

1. **几何/映射错**：`#247` 面板↔CCD 映射至 11-23 仍未冻结（11-23 波形判别 verdict
   `patch_position_specific`；axis_gate fail：blaze 两轴夹角 0.26° vs 期望 88.544±2°；tilt_x vs tilt_y
   仅差 0.46°–0.63°）。治理条款 `next_experiment_plan_and_capture_20261120.md` §5.3 明确：
   **映射未冻结前，方向门 fail 不能归因于模型本身**。⇒ 11-20 的 4/14 目前**不可归因**。
   - 2026-11-23 轴路由离线重判（零设备，非预注册）已把上述「x 族路由异常」的**适用范围钉窄**：
     **11-20 transfer 门控数据本身无轴异常**——12/12 ramp 标定行 ramp_x→row、ramp_y→col
     与模型一致（幅度比 0.993–1.007、角 ~179.5° 属 phase_sign 张力非轴交换）；V1 实测转置 /
     V2 模型转置 / V3 180° / V4 转置+取负 / V5 转置+平移 全部 0/14（V5 1/14，且为自由平移
     搜索伪影，被跨模式无关对 null 对照 51/84 否定）；fy≡0 前提类在被探测模式中为空
     （min|row/col| = 0.600，Z8；Z5–Z11 均非行恒定图案）；强模式位移方向 meas/model 一致
     （0.5–10°，无 ~90° 特征）。4/14 的更可能成因：**~8 px 原点/配准偏移**（拟合原点
     (1235.08, 1104.64) vs 门控椭圆中心 (1227, 1104/1103)；改用拟合原点重配准 4/14→7/14、
     仍 <10/14）+ 差分形状/幅度失配（meas 幅度 3.9–8.6×、PSF 更宽）+ Z7 低信噪比（两 pass
     复现性 −0.315）+ phase_sign 张力（见 §6）。**x 族落列轴只在 11-21/11-23 issue247 采集路径
     成立**（grating_x/tilt_x → CCD 列 ±95.6 px）⇒ 路由行为**路径相关**，两链约定表核对
     （T5–T8）进行中，S1b 图样集待其定稿。
2. **模型算子错**：需在映射闭合后才可判。
3. **测量无分辨力**：已做可分辨性审计（`scripts/slm/audit_direction_gate_resolvability_20261123.py`）：
   pass1↔pass2 实测差分 Pearson = Z5 .927 / Z6 .857 / **Z7 −0.315** / Z8 .976 / Z9 .784 / Z10 .893 /
   Z11 .853 ⇒ **6/7 模可分辨**（Z7 两 pass 反号，不可分辨，需第 3 pass 判决）。
4. **强度门判决无效**：11-20 为 Mono8（违反 Mono12 纪律）；8-bit 预言机地板逐 holdout 0.191–0.315、
   **中位 0.244 > δ=0.20**（`scripts/slm/audit_strength_gate_floor_20261123.py`）⇒ 该批数据上
   连完美预测者都过不了强度门 ⇒ 11-20 强度门降级 **not_assessable**，**δ=0.20 不放宽**。
5. **过门率对极性/几何口径极度敏感**（`scripts/slm/audit_p3_finite_difference_20261123.py`）：
   78 行大样本（Z4–Z11 × {0.10, 0.20}）方向门 pooled **2/16**（0.10 档 2/8、0.20 档 0/8；
   只看可释放的 Z5–Z11 为 **1/14**），比 14 对小样本 4/14 **更差**（差异主要来自 0.20 档与 Z8）。
   口径变体：`phase_sign=+1` → **0/16**；残差取反 → **0/16**；生产拟合几何 → **0/16**。
   而 `p3_fit_20261120/frozen_parameters.json` **没有 `phase_sign` 字段**（默认 +1），门协议却用 −1
   （#235 实测极性）⇒ **S3 必须先冻结极性/残差/几何三项约定并同时报敏感性**，否则判决不可复现。
6. **P3-1 有限差分 Jacobian**（同一脚本）：可辨识率 **0.143**（1/7，仅 Z11）、列相关 cond **48.8**；
   同一 P0 配方在本台复算得 **50.4 / 0.143** ⇒ P0 的 `0.0 / 17425` 主要是**几何口径产物**，
   不能与新值相减、也不能据此宣称「可辨识性变好」。实测侧 6/7 列两两耦合 **|r| ≥ 0.76–0.92**
   （Z5↔Z9、Z6↔Z8、Z10↔Z5 最强），对角/非对角比 0.88/0.80 < 3 ⇒ 矩阵门不过。

---

## 1. 门控定义与执行顺序

| 门 | 判据（数值化） | 门控对象 |
|---|---|---|
| **G1 映射闭合** | ① A 映射残差 ≤1.0 px（2D blaze2d / 局部族 / 1D y 族三类分别给残差）；② 各轴 1/p 位移拟合残差 ≤1.5 px；③ 搜索 ROI **必须由预测位置导出**（禁止 fixed `core_exclude`）；④ 内容相关判决（S1b）：内容 A（逐行相同）落列轴、内容 B（扰动一行）落行轴 | S3 方向门重判、S4、P3-2 重启 |
| **G2 测量地板** | 同图样重复帧 oracle relL2 **中位 ≤0.10**（= δ/2） | **仅**强度门判决；不满足 ⇒ 强度门记 `not_assessable`。方向门是 Pearson 相对量，不受 G2 门控 |
| **G3 数据纪律** | Mono12（`bit_depth=12` / `effective_full_scale=4095` 双重确认）+ 同曝光暗场 + 全幅不物理开窗 + 饱和即降曝光重拍该档 | 所有判决数据 |
| **G4 零级 PSF 自洽** | 零级 FWHM **≤25 px** + 四口径质心差 **≤3.0 px** + 离焦 D 本窗口重新登记 | S1 拟合、S3 参考帧 |

**执行顺序（一次 open/close，段间冷却 ≥200 s）**：

```
S0 光路探测(G3,G4) → S1 映射闭合(G1) → [G1 不过 ⇒ 保留 S2 载荷后终止窗口] → S2 辐射/非线性+地板(G2)
   → [G2 记 not_assessable 时强度门不出判决] → S3 方向门重判(G1+G3+G4 齐备) → S4 条件段
```

---

## 2. 段表（预注册判据，采集前冻结）

| 顺序 | 段 | 采什么 | 帧数 | 位深/曝光 | 预注册数值判据 | 失败退出 |
|---|---|---|---|---|---|---|
| **S0** | 光路六步探测（含漂移基线、奇偶基线、焦面扫描、离焦 D 重测） | 零级 1、四方向 4、漂移对 2、奇偶对 2、焦面扫描 3 | 12 | Mono12，全幅，0.087 ms（饱和即降曝重拍） | ① `probe_status=ok`；② `serial_source=="driveboard"`；③ 零级 FWHM ≤25 px；④ 零级可定位；⑤ 无饱和；⑥ 落盘 `optical_path_record.json`（含新 D） | 零级不可定位或反复饱和 ⇒ **终止本窗口**（S0 数据保留），修光路后重排 |
| **S1** | **映射闭合 + 内容相关性判别**，拆三小段（见 §2.1） | S1a：A 映射闭合族（1D 全幅 y 族 3 图样 ×±1 = 6；局部族 patch_x/patch_y/ladder ×± = 6；全幅 2D blaze2d ± = 2；2-D 正交格点 1；同曝光暗场 1）= 16 帧。S1b：内容相关探针 4 帧 | 20 | Mono12，全幅，0.087 ms | ① **S1a**：A 映射残差 ≤1.0 px（2D blaze2d / 局部族 / 1D y 族三类分别给残差）；② 1/p 位移拟合残差 ≤1.5 px；③ ROI 由预测位置导出；④ **S1b 内容相关判决**（见 §2.1）：内容 A（逐行相同）落列轴、内容 B（扰动一行）落行轴 ⇒ 内容相关路径确认；⑤ 两轴夹角落档（2D 期望 88.544±2°；≈0° ⇒ 共线实锤） | S1a ① 不过 ⇒ A 映射不可冻结 ⇒ 保留 S2 载荷后终止窗口；S1b ② 不过 ⇒ 异常非内容相关，x 族映射口径需重查（与 blaze2d 正常相矛盾），登记待查 |
| **S2** | 辐射/非线性（#233 载荷）+ 测量地板审计 | 线arity 3 幅度 3 帧；同图样重复 2 帧（oracle 地板）；同曝光暗场 1 | 6 | Mono12，全幅 | ① 强度–幅度线性拟合 relL2 ≤5%；② 重复帧 oracle relL2 中位 ≤0.10 ⇒ **G2 过** | G2 不过**不终止**，仅使 S3 强度判决记 `not_assessable` |
| **S3** | 方向门重判 | Z5–Z11（7 模）× ±0.05 rad × **3 个独立 pass** + 零级参考 3 帧 | 45 | Mono12，全幅，0.087 ms | ① Pearson ≥0.50/对；**主判 pooled ≥10/14（预注册不变）**；② 同时报告剔除 Z7 的 12 对（敏感性，不改主判）；③ 每对 SNR `rms(Δ)/mean|anchor| ≥ 0.02`（不达记 `insufficient_signal`，**不计入分母**）；④ 强度门仅在 G2 过时判（worst-pass relL2 ≤0.20）；⑤ **极性/几何三项约定在采集前书面冻结**（`phase_sign` 与作用域、残差符号、拟合几何：checkpoint 采样网格 vs 生产 `build_fit_model` 几何），并**同时报告三种口径的过门数**（现协议 −1 / checkpoint 默认 +1 / 生产拟合几何） | ① 不满足 ⇒ 模型方向门 **fail（映射已闭合 ⇒ 归因有效）**，不执行 S4；Z7 第 3 pass 仍反号 ⇒ 触发同根因下游排查（零设备优先） |
| **S4** | patch 位置相位判别（条件段） | 同一 blaze patch（period 96 px）中心 3 位（panel 中心 / +256 px panel-x / +256 px panel-y）× 2 | 6 | Mono12，全幅 | 多 patch 间映射位移差 ≤1.0 px ⇒ 位置无关成立；>1.0 px ⇒ 确认位置相关，登记逐 patch 映射 | 条件段：仅当 S3 处于 10–11/14 边缘、或方向门过而强度门边缘时执行 |

### 2.1 S1 三小段（要点）

**S1a —— A 映射闭合族（16 帧）**：只用**已知走 A 映射**的内容类来闭合 A 映射。

| 内容类 | 图样 | 帧数 | 判读 |
|---|---|---|---|
| 全幅 1D y 族（fy≠0） | grating_y p128 ±、p256 ±、ramp_y ± | 6 | 幅值 = K·N（N=1200/Λ）、落列轴，残差 ≤1.5 px |
| 局部族 | blaze_patch_x ±、blaze_patch_y ±、ladder ± | 6 | 走 A 映射（x→行、y→列），残差 ≤1.0 px |
| 全幅 2D | blaze2d ±（双轴周期 128/192、192/128 两组中取一组） | 2 | 四斑实测 vs A 映射预测残差 ≤1.0 px（11-21b 复核基线 ≤1.8 px） |
| 2-D 正交格点 | 同行同列梯度共存的连续 blaze 格点，± 反号 | 1 | **单帧同时读出两条基向量**，判两轴真实夹角（期望 88.544±2°） |
| 暗场 | 同曝光 | 1 | G3 前置 |

**S1b —— 内容相关性判别（4 帧，本窗口的决定性实验）**：同一 15 周期 / 7.5 周期 fx 图样，
两种内容各下 ±1 级：

- **内容 A（逐行完全相同，fy≡0）**：与 11-23 写法相同。**预期**：落 CCD 列轴，幅值 = K_col·N
  （N=1920/Λ=15 或 7.5 ⇒ ≈96 px / ≈48 px），残差 ≤1.5 px。
- **内容 B（同图样 + 扰动一行）**：在内容 A 上叠加「第 0 行加一个像素级相位阶跃」，其余行不变。
  **预期**：若恢复**行轴**、幅值 = K_row·N（≈93 px / ≈47 px），则**内容相关路径（SDK/固件或面板内部
  10-bit 存储）实锤**；若仍落列轴，则异常与内容无关（与 blaze2d 正常相矛盾），登记待查。

**通用规则**：

- **离焦补偿**：按 10-19 实测 D = +0.559 m⁻¹ 叠面板相位；**D 必须本窗口重测**（不得假设仍有效）。
- **ROI**：一律由预测位置导出（1/p 律：x 族 p128 ≈95.6 px、p256 ≈47.8 px；y 族 p128 ≈59.6 px、p256 ≈29.4 px），
  **禁用** 11-23 分析里那个 fixed `core_exclude=34.96 px` —— 它恰好把 y 族 p256 的真 ±1（29.4 px）排除掉，
  导致 11-23 把 **±3 级（88.89 px）**误认成主瓣。
- **全幅 1D x 族（fy≡0）不参与 A 映射校验**：其真值语义为「列轴 + 1920/1200 内容压缩」，单独记录。

---

## 3. 设备纪律（逐条执行）

- **一次窗口一次 open/close**；S1–S4 同进程串行，段间冷却 **≥200 s**；禁止背对背 open/close（驱动板假死 `ReadSU=-1000`）。
- **Mono12**（8-bit 下 1 count ≈7%，均匀度/强度结论无效）；全幅采集**不做物理开窗**。
- 任一帧峰值 ≥0.9×满量程 ⇒ 当档降曝光**重拍该档**，不裁数据。
- 起始 `--wait-until-idle 600`；每段独立 `--out`（目录不可复用，`mkdir(exist_ok=False)`）。
- 采集期只做采集，**判读全部离线**；USB `-1000/-10002` ⇒ 当日停手，不 reopen。
- 采集后每段必须落 `released.json`（`slm_closed/camera_closed/lock_released`）与窗口 `window.json`。

---

## 4. 预算

| 情形 | 帧数 | 墙钟 |
|---|---|---|
| 基线（S0–S4 全跑） | **89**（12+20+6+45+6） | ≤95 min（起始 idle 10 min + 采集 ~60 min + 4×200 s 冷却 ~13 min） |
| G1 不开（S0–S1 后终止，保留 S2 载荷） | 38 | ≤55 min |
| 含可选 #234 实验 A/B 上限 | 95 | ≤100 min |

---

## 5. 相对既有数据的新增量（每段）

| 段 | 新增量 |
|---|---|
| S0 | 本光路首次标准 `optical_path_record.json`（11-23 只有波形 verdict）；漂移/奇偶基线从无到有；D 重测（10-19 的 +0.559 m⁻¹ 不得假设仍有效） |
| S1 | ① **首次**用「已知走 A 映射」的内容类（2D blaze2d / 局部 patch·ladder / 1D y 族）闭合 A 映射并给残差；② **首次**做内容相关判别（同 fx 图样：逐行相同 vs 扰动一行），这是把映射异常定级到「SDK/固件/面板内部存储」还是「映射口径」的唯一判别实验；③ 2-D 正交格点首次直接量两轴角（此前只有单族推断 0.26°）；④ ROI 由预测位置导出（修正 11-23 把 ±3 级误认主瓣的假象）；⑤ 逐图样记录实际写入灰度字节 SHA256 + 相位哈希（供 #235 byte-identity 对账） |
| S2 | 首次「线arity + 同曝光暗场」联采（11-20 无暗场、Mono8）；12-bit 重复帧地板替代 8-bit 地板 0.244 |
| S3 | 3 pass（此前 2）+ Mono12 + Z7 可重复性判决；把 4/14（不可归因）变成**映射闭合下的有效重判** |
| S4 | 多 patch 位置定量（11-23 仅 verdict，无多位置数据） |

---

## 6. 判决口径与边界（预注册）

- **方向门**：Pearson ≥0.50/对，pooled ≥10/14（主判，不改）；剔除 Z7 的 12 对只作敏感性报告。
- **强度门**：worst-pass masked absolute relL2 ≤0.20（δ 不放宽）；**仅当 G2 过**才出判决，否则 `not_assessable`。
- **`insufficient_signal` 不计入分母**（每对 SNR `rms(Δ)/mean|anchor| ≥0.02`）。
  注：现有 `scripts/slm/direction_holdout_gate_20261119.py` 对 NaN 相关性记 `passed=False` 并计入分母，
  与本口径不一致 → 本窗口判读须用修正口径（离线脚本，不改已归档 JSON）。
- **归因边界**：只有 G1 闭合后 S3 的方向门 fail 才可写「模型算子错」；否则只能写「不可归因」。
  本窗口**不产出**任何标定（LUT / 效率 / 映射标定值），只产出判别结论。
- **内容类必须记录**：S3 每对须记录该 Zernike 模式的 **fy≡0 属性**（行方向梯度能量占比）与
  实测差分位移轴；方向门按内容类（fy≡0 / fy≠0）**分别报**，不合并成单一 pooled 数。
  判读规则：若 fy≡0 类整体不过门、fy≠0 类过门 ⇒ 判为**映射路由问题**（转 #247），
  **不是模型算子问题**；若两类都不过门 ⇒ 才轮到模型算子/PSF 形态。
  2026-11-23 轴路由复核注记：Z5–Z11 的 row/col 梯度能量比实测 0.600–1.667，**无一为
  fy≡0 类**（该类在 Noll 组中仅 Z2/Z3 纯 tilt）⇒ 本条分支在本窗口 S3 上预计**退化为全
  fy≠0 单类**；条款保留（实测时逐对记录即可），但「fy≡0 类不过门 ⇒ 路由问题」的分支
  在 S3 上大概率无样本，路由判决的真正落点在 S1b 内容相关判别。
- **事后口径修正必须标注**：任何「按修正映射（或按内容类分支）重判」的结果属**非预注册**，
  只能作为诊断证据，不得当作原预注册方向门的通过结论。
- **口径冻结条款**：`phase_sign`（含作用域）、残差符号、拟合几何三项在 S3 采集前书面冻结并写入
  `optical_path_record.json`；11-20 数据显示这三项各自都能把过门数从 2/16 压到 0/16
  （且 checkpoint `frozen_parameters.json` 无 `phase_sign` 字段，默认 +1 与门协议 −1 不一致）
  ⇒ **不冻结口径的方向门判决不可复现、不可引用**。

---

## 7. 落盘约定

- 采集：`data/experiments/<experiment>/<window>/<segment>/`（`data/*` 已 gitignored）。
  每段独立目录 + `released.json`；窗口级 `window_plan.json`、`segments/<name>.{json,log}`、`window.json`。
- 判读产物：`data/debug/slm_transfer/<window>/`（JSON/MD）+ 报告落 `report/shape_trials/<window>/`，
  按 [report/AGENTS.md](../../report/AGENTS.md) 登记 provenance。
- 窗口信封：`scripts/experiments/_window.py::add_window_args()` 提供 `--cooldown-s`(默认 200)、
  `--wait-until-idle`(默认 600)、`--segment-timeout-s`、`--execute`，但**本预注册没有任何窗口脚本实例化它**
  （现只有 `run_transfer_window_20261119.py` / `run_issue238_period_sweep_20261119.py` 在用）⇒ 本窗口按 §7.1
  逐段手敲时，段间冷却与 open/close 纪律**没有代码背书，只能由人保证**（见 §3）。

### 7.1 可执行命令行（2026-11-23 零设备核对 argparse 后补写）

> 本节在 2026-11-23 由「预注册只有散文、无可执行命令」这一缺陷补写。所有参数名/默认值均核对过脚本
> `add_argument`，**未运行任何 `--execute`**。段间一条命令一条命令手敲时，冷却与 open/close 纪律由人保证。

**共同前提**

- `data/experiments/attribution_20261123/<segment>/` 每段**必须不存在**：`issue247_geometry_capture` /
  `efficiency_match_capture` 用 `out.mkdir(exist_ok=False)`，`transfer/capture.py:551` 同款 ⇒ 目录已存在即**失败**（不复用）。
- 一个窗口一次 open/close，**段间人工等待 ≥200 s**；全部段 `--wait-until-idle 600`、曝光 0.087 ms、Mono12。
- 各采集脚本内部都是「恰好一次 open / 一次 close」（`issue247_geometry_capture_20261120.py:49-63`、
  `efficiency_match_capture_20261123.py:45-51`）⇒ 手敲多段等于多轮 open/close，冷却只能靠人间隔。

**S0 — 光路六步探测**

> ⚠ **该脚本没有 `--execute` 门控**（设计如此，见 `tests/ao_shaping/scripts/test_device_lock_callers.py`：
> 它必须整段持锁）：**敲回车即占用 SLM+CCD**。`--wait-until-idle` 默认 **0.0 = 不等、立刻报错** ⇒ 必须显式 600。
> 无 `--out`，只有 `--out-root` + `--session`；实际落盘 `<out-root>/<session>/<UTC戳>/`，事后须把该路径写进窗口记录。
> 曝光参数是**逗号分隔字符串** `--exposures`，不能写成 `--exposures 0.087 0.2`。

```powershell
.venv/Scripts/python.exe scripts/slm/probe_optical_path_validity.py `
  --bit-depth 12 --wait-until-idle 600 --exposures 0.087 `
  --session s0_attribution_20261123 `
  --out-root data/experiments/attribution_20261123/s0
```

**S1a — A 映射闭合（4 图样 × 4 帧 = 16 帧）**

> 位深**硬编码** `BIT_DEPTH_REQUIRED=12` / `FULL_SCALE_REQUIRED=4095`（`:194-195`），开相机后回读不符即报错；
> `--out` 非 self-test 时必填且必须不存在（`:1414`）；`--wait-until-idle` 默认 **1800**（`:192`）；
> 每图样固定 4 帧（无 `--frames-per-pattern` 旋钮）⇒ 只能用 `--patterns` 选图样控帧数。
> 图样 id 见 `:171-187`（waveform）与 `--plan frozen` 的 15 项（`:962-980`）。1D y 族的 `grating_y_p256`
> 放在 S1b 子集里取（同一脚本、同一窗口），避免 A 映射段被 y 族误差污染。

```powershell
.venv/Scripts/python.exe scripts/slm/issue247_geometry_capture_20261120.py `
  --execute --out data/experiments/attribution_20261123/s1a `
  --plan frozen --exposure-ms 0.087 --wait-until-idle 600 `
  --patterns flat,ladder_x,ladder_y,blaze2d_px128_py192_sx+1_sy+1
```

**S1b — 内容相关判别（图样子集待 2026-11-23 轴路由离线复核结果定稿）**

> 能力核对：`--plan waveform` 含 `grating_x_p256`（逐行相同，fy≡0）与 `grating_y_p256`
> （逐列相同，fx≡0，同周期）与 `blaze_patch_x/y`（局部、非行恒定）⇒ **决定性 A/B 已在现有图样表内，
> 不需要新脚本**，也不需要「扰动一行」的构造器。`--patterns` 传未知 id 会直接报错（`:1742-1746`）。

**S3 — 方向门重判（3 pass）**

> 位深缺陷已修（2026-11-23）：`direction_holdout_capture_20261119.py` 原**无位深参数**，经
> `transfer/capture.py:299-301` 只在 `args.bit_depth` 存在时才下发 ⇒ 该脚本此前一律落 **Mono8**。
> 已加 `--bit-depth`（默认 12、`choices=(8,12,16)`）并写入传给 capture 的 namespace，由
> `_verify_capture_bit_depth` 回读校验。`--out` required、`exist_ok=False`；`--wait-until-idle` 默认 **120**。
> 帧数实测（plan-only）：每 pass 14 个 holdout 探针 + 标定段（reference_start / ramps(64,128) / reference_end），
> `--frames 2` ⇒ **约 36 帧/pass**，3 pass ≈ 108 帧（原 §2 写的 45 帧按 `--frames 1` 估，与实际不符）。
> 若要压到 45 帧级：pass1/pass2 全 7 模、pass3 只跑 Z7（4 帧），并把 `--frames` 降到 1。

```powershell
.venv/Scripts/python.exe scripts/slm/direction_holdout_capture_20261119.py `
  --execute --out data/experiments/attribution_20261123/s3_pass1 `
  --modes 5 6 7 8 9 10 11 --amp 0.05 --frames 2 --bit-depth 12 `
  --exposure-ms 0.087 --wait-until-idle 600
# pass2 同上，仅 --out 换成 s3_pass2；pass3 同（若只判 Z7 则 --modes 7）
```

**S3 判读（离线，不占设备）**

> `--out` 这里是**JSON 文件路径**（不是目录），且 `args.out.parent` 用 `exist_ok=True` ⇒ 同名文件会被覆盖。
> `--runs` 与 `--checkpoint` 必填；`--center-method` 只有 `ellipse` 一个 choice。
> ⚠ 该脚本对 NaN 相关性记 `passed=False` 并计入分母，与 §6 口径冲突（判读须走修正口径）。

```powershell
.venv/Scripts/python.exe scripts/slm/direction_holdout_gate_20261119.py `
  --runs data/experiments/attribution_20261123/s3_pass1 `
         data/experiments/attribution_20261123/s3_pass2 `
  --checkpoint data/debug/slm_transfer/p3_fit_20261120/frozen_parameters.json `
  --out data/debug/slm_transfer/attribution_20261123/s3_direction.json
```

**S4 — patch 位置相位（条件段）**

> 同一脚本；`blaze_patch_x/y` 的 origin 是模块常量（`:137`、`:397-400`），**没有「中心 +256 px」的 CLI 旋钮**
> ⇒ 3 个位置要么改常量要么另写脚本，本窗口暂按「单位置 × 2」降级执行。

```powershell
.venv/Scripts/python.exe scripts/slm/issue247_geometry_capture_20261120.py `
  --execute --out data/experiments/attribution_20261123/s4 `
  --plan waveform --exposure-ms 0.087 --wait-until-idle 600 `
  --patterns blaze_patch_x,blaze_patch_y
```

**S2 — 范围待改（见本节末 S2 条）**

> `efficiency_match_capture_20261123.py` 固定 16 帧（flat6/blaze4/shaping4/dark2，`:99`、`:109`）、
> **无 `--wait-until-idle`、`--cooldown-s` 默认 20**（`:126`）⇒ 不能承 §2 的「3 幅度线性 + 2 重复 + 1 暗场」。

---

## 8. 变更协议

- 本文件在**采集开始后不得回改判据**；若必须改阈值/判据，另起带日期的新文件并写明原因。
- 每次窗口结束后逐项回写 issue（#233/#234/#235/#247）并附 commit id。

---

## 9. 相关文件

- 状态入口：`docs/dev/hw_plan_consolidated_20261118.md`（§2 P3/P4 行、§3 G1–G4、§4 S5/S6 暂缓、§7.4）
- 11-20 冻结稿（归因条款）：`docs/dev/next_experiment_plan_and_capture_20261120.md` §5.3
- 方向门/强度门脚本：`scripts/slm/direction_holdout_gate_20261119.py`、
  `scripts/slm/evaluate_strength_gate_20261119.py`
- 门审计（零设备，2026-11-23）：`scripts/slm/audit_direction_gate_resolvability_20261123.py`、
  `scripts/slm/audit_strength_gate_floor_20261123.py`；产物
  `data/debug/slm_transfer/p3_gate_20261120/{resolvability,strength_floor}_audit.json`
- #247 波形判别：`data/debug/hw_window_20261123_issue247_waveform_probe/WAVEFORM_PROBE_REPORT.md`
- 轴路由离线重判（2026-11-23，零设备）：11-20 门控数据无轴异常（V1–V5 全 0/14、
  12/12 ramp 轴向正确、fy≡0 类为空）；报告与中间产物
  `C:/Users/zhangh/AppData/Local/Temp/opencode/issue247_axis_report_20261123.md`、
  `issue247_axis_*.{json,txt}`（结论已并入 §0.2 与本条，双链约定核对 T5–T8 进行中）
- 原子计划：`docs/dev/hw_direction_test_plan_20261117.md`（P1–P5）
