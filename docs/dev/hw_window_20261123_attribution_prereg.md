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
     （T5–T8 已定稿：T8 改判为「内容类相关」，仅 issue247 链「全幅 fy≡0 一维 x 族」呈现差异，
     触发条件未定级，#247 不冻结），S1b 采用下方 §「S1b」的 2×2 factorial 设计（内容 A/B × fx
     周期 15/7.5）判别。
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
| **S3** | 方向门重判 | Z5–Z11（7 模）× ±0.05 rad × **3 个独立 pass** + 零级参考 3 帧 | 45 | Mono12，全幅，0.087 ms | ① Pearson ≥0.50/对；**主判 pooled ≥10/14（预注册不变）**；② 同时报告剔除 Z7 的 12 对（敏感性，不改主判）；③ 每对 SNR `rms(Δ)/mean\|anchor\| ≥ 0.02`（不达记 `insufficient_signal`，**不计入分母**）；④ 强度门仅在 G2 过时判（worst-pass relL2 ≤0.20）；⑤ **极性/几何三项约定在采集前书面冻结**（`phase_sign` 与作用域、残差符号、拟合几何：checkpoint 采样网格 vs 生产 `build_fit_model` 几何），并**同时报告三种口径的过门数**（现协议 −1 / checkpoint 默认 +1 / 生产拟合几何） | ① 不满足 ⇒ 模型方向门 **fail（映射已闭合 ⇒ 归因有效）**，不执行 S4；Z7 第 3 pass 仍反号 ⇒ 触发同根因下游排查（零设备优先） |
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

**S1b —— 内容相关性判别（4 帧，2×2 因子设计，本窗口的决定性实验）**：
因子 1 = **内容**（A：逐行完全相同，fy≡0 / B：A + 扰动一行）；因子 2 = **fx 周期**
（Λ=128 px，15 周期 / Λ=256 px，7.5 周期）。每格 1 帧、共 4 帧，采集顺序开窗前冻结。
因子 2 不是待估参数而是**内部复现**：轴归属若真由内容决定，应在两个空间频率上一致；
「两周期同时满足 K·N 律」正是防把 ±2/±3 级或杂散误判为主瓣的复现——11-23 已犯过一次
（±3 级 88.89 px 被误认主瓣，真 ±1 在 29.4 px）。这也是选 2×2 而非 2×1 的理由。

| 格 | Λ=128（N=1920/128=15） | Λ=256（N=7.5） |
|---|---|---|
| **内容 A**（fy≡0，与 11-23 写法相同） | 预期**列轴**，K_col·N ≈96 px，残差 ≤1.5 px | 预期**列轴**，K_col·N ≈48 px，残差 ≤1.5 px |
| **内容 B**（A + 第 0 行像素级相位阶跃，其余行不变） | 内容相关 ⇒ **行轴** K_row·N ≈93 px；内容无关 ⇒ 列轴 ≈96 px | 内容相关 ⇒ **行轴** ≈47 px；内容无关 ⇒ 列轴 ≈48 px |

- 判读主统计量（每格）：主瓣**轴归属**（行/列）+ 幅值残差 |measured − K_axis·N|（判 ≤1.5 px）。
- **内容相关实锤（4/4 一致）**：内容 B 两格**一致落行轴**（K_row·N，残差 ≤1.5 px）**且**
  内容 A 两格**一致落列轴**（K_col·N，残差 ≤1.5 px）⇒ 内容相关路径（SDK/固件或面板内部
  10-bit 存储）确认。
- **内容无关**：B 与 A 同轴同幅值（K_col·N）⇒ 异常与内容无关（与 blaze2d 正常相矛盾），登记待查。
- **不一致（`insufficient_consistency`）**：B 两格轴归属不一致，或任一格幅值不服从 K·N 律
  ⇒ 登记待查、不做实锤；单格异常不得外推。

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
| 基线实测校正（2026-11-23 plan-only 复核） | **见 §7.1 S3 条**：`--frames 2` 下每 pass **36 帧**（14 探针 ×2 帧 + 标定段 8），3 pass = **108 帧**（原 45 未计标定段，按 `--frames 1` 高估）；`--frames 1` + pass3 只跑 Z7 ⇒ S3 **54 帧**（22+22+10）、总 **98 帧** | `--frames 2` ⇒ 总 **152 帧**（+63 vs 基线 89），墙钟 ≈2.1 h，**须重新批**；`--frames 1` + Z7-only ⇒ 98 帧 ≈89 min，**勉强在 ≤95 min 预算内** |
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
```**S3 判读（离线，不占设备）**

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

---

## 10. 修订 R1（2026-11-24）：S0 补测 Fresnel D —— 指令与判据

> 本节是**附加段**：§2 的段表、§6 的判决口径**一字不改**。它只解决 §2 S0 判据 ⑥「落盘
> `optical_path_record.json`（含新 D）」为何未闭合，以及在设备已上线的前提下如何闭合。

### 10.1 触发事实（2026-11-24 10:42 本地窗口实测，非推测）

| 事实 | 证据 |
|---|---|
| S0 六步探测已通过 | `data/experiments/attribution_20261123/s0/s0_attribution_20261123_lowexp/20261124_024252Z/optical_path_probe.json` 的 `probe_status=ok`（0.05 ms、峰值 2637/4095、噪声底 20.9、无饱和） |
| 但 D 仍是 `null` | 同目录 `optical_path_record.json` 的 `optical_path.fresnel_compensation_m_inv = null`，理由字段写着「探针不做 Fresnel 补偿标定（compensation 由焦面标定给出）」 |
| 探针结构上测不出 D | 该段只做 Noll-4（rad）离焦扫描，判的是「离焦响应是否显著」，**与 Fresnel 光焦度（m⁻¹）不是同一量**，不能换算 |

⇒ **S0 未闭合，S1 不得开始**（§1 执行顺序 + 判据 ⑥）。本节定义补测段 **S0‑R1**。

### 10.2 实现落点（为什么改探针而不是另写脚本）

在 `scripts/slm/probe_optical_path_validity.py` 内新增**默认关闭**的 `--focus-calibration` 段，
在该脚本既有的「一次 open/close、持锁到底、位深回读校验、放锁后才写 `released.json`」纪律内
调用 `src/ao_shaping/tools/slm/panel/focus_probe.py::sweep_focus`（设备无关的焦面扫描，
自带逐点自适应曝光、`peak/exposure` 排序、抛物线顶点四点复测、曝光线性度检查）。

- 不新增脚本：设备锁、位深校验、`occupancy.released_at`、`--wait-until-idle`、`--out` 契约都已在探针里，
  另写一份就等于复制一遍纪律并留下两套要同步的代码。
- **不默认开启**：不带 `--focus-calibration` 时行为与 2026-11-24 02:42Z 那次逐位一致（D 仍 `null`、理由保留）。
- 同一次会话内、现有各段之后、**放锁之前**执行 ⇒ 不产生第二次 open/close（§3 禁止背靠背）。

### 10.3 命令契约（S0‑R1）

> 该脚本**无 `--execute` 门控**（必须整段持锁，见 `tests/ao_shaping/scripts/test_device_lock_callers.py`）：
> 敲回车即占用 SLM #1 与 CCD。`--wait-until-idle` 默认 0.0＝不等，必须显式 600。
> 曝光是逗号分隔字符串。`--out-root` + `--session` 决定落盘 `<out-root>/<session>/<UTC戳>/`。

```powershell
.venv/Scripts/python.exe scripts/slm/probe_optical_path_validity.py `
  --bit-depth 12 --wait-until-idle 600 --exposures 0.05 `
  --session s0r1_dcal_20261124 `
  --out-root data/experiments/attribution_20261123/s0 `
  --focus-calibration `
  --focus-centre-px 960,600 --focus-radius-px 450 `
  --focus-coarse-points 13 --focus-diopter-range 1.5 `
  --focus-refine-points 7 --focus-frames 3 `
  --focus-floor-ms 0.02 --focus-ceiling-ms 20
```

- `--focus-centre-px 960,600 --focus-radius-px 450`：**冻结值，SLM 面板坐标（panel px）**。
  `focus_probe.py::_measure_point`（292–320 行）把这两个数交给 `fresnel_panel`，用作 Fresnel
  透镜的 `pupil_center` 与 `radius`；**不是相机帧坐标**，也不是相机里 0 级的 argmax。
  二者与 02:42Z S0 的 `config.pupil_center_px=[960,600]`、`config.defocus_radius_px=450`
  完全一致，且与 `bench_kernels.BEAM_CENTER_PANEL/BEAM_RADIUS_PANEL` 的默认值一致。
  相机帧里的 0 级 argmax 实测 ≈(1227.7, 1103.2)（10 帧 x=1227–1229、y=1102–1105），
  **只用于 S1 的零级定位，绝不传给 D 标定** —— 两者坐标空间不同，混用必错。
- `--focus-floor-ms 0.02`：0.05 ms 那次 `optical_path_probe.json` 的 `cam_exposure_range_ms=[0.02, 1000.0]`，
  **0.02 是本机设备下限**（不是随便写的 0.011）。最佳焦点的中心像素远亮于离焦面（离焦 0.05 ms 已到
  2637/4095），自适应曝光会被压到下限；**若在下限仍饱和**，按 10.4 D4 记为失败，不加衰减片、不外推。
- 参数（13 点 / ±1.5 m⁻¹ / 步长 0.25 m⁻¹ + 7 点细化 + 3 帧平均）**在本节冻结**，与历史值无关地独立选取。

### 10.4 判据（采集前冻结）

| 编号 | 判据 | 不满足时 |
|---|---|---|
| **D0 SLM 相位自检** | 调用 `display_state.slm_self_check(slm)`：`get_current_grayscale()` 必须为 0；`get_displayed_memory_number()` 的 code=1 属正常；回读结果入证据 | 中止 D 标定段、不写 D，记录失败原因并**人工提醒用户**检查面板状态后重排；不连带作废既有 12 帧 S0 结果 |
| **D1 内点** | 最优点必须是不在扫描端点的内点（左右各有 ≥2 个采样点） | 落端点 ⇒ 允许自动外扩一次；外扩后仍落端点 ⇒ 标定失败，**记 `out_of_range`，不外推** |
| **D2 单峰** | 最优点两侧的 `peak_per_ms` 至少各单调下降 2 个采样点 | 记 `ambiguous`，**不写 D** |
| **D3 曝光线性度** | `sweep_focus` 自带线性度检查通过（`peak/exposure` 排序成立） | **不写 D**，把失败证据留档 |
| **D4 不饱和** | 最优点及复测点在 `effective_full_scale=4095` 下无饱和像素 | 记 `saturated_at_floor`，**不写 D**；下一步是加衰减片重测（另起修订） |
| **D5 可复算** | `focus_calibration.json` 含全部扫描点（diopter / exposure / peak）与复测点，任何人可离线复算出同一个 D | 缺证据 ⇒ D 不写 |

**写入规则**：只有 D0–D5 全过，才把实测值写进 `optical_path_record.json` 的
`optical_path.fresnel_compensation_m_inv`，并删除「探针不做 Fresnel 补偿标定」那条理由；
同时新增 `fresnel_calibration` 证据子对象（run_id、UTC 时间、方法、`centre_px` 与其来源、
`radius_px`、网格参数、逐点读数、复测点、status、SLM/相机序列号）。
任何一项不过 ⇒ 字段保持 `null`，理由改写为具体失败原因。**可疑值不得当 D 记录。**

### 10.5 旧值声明（重要）

本窗口**不引用任何历史 D**。理由：预注册旧值 `+0.559 m⁻¹`（§5 S0 行）**找不到出处**；
`report/shape_trials/hw_defocus_zernike/artifacts/*/focus_calibration/slm1_*.json` 里的历史值互不相等
（`+0.626 / +0.645 / +0.652 / +0.739 / +0.746`），且部分文件**内部字段自相矛盾**。
因此历史值既不能作为新 D 的等价性基准，也不能用来做范围一致性检查 —— 新 D 独立成立。

### 10.6 帧数与预算

焦面扫描的帧数由扫描点决定，**不并入 §2 S0 的「12 帧」**（那 12 帧是六步探测的）。按本节参数：
13 个粗扫点 + 最多 1 次外扩（+6）= 19 个粗扫点；再加 7 个细化点、1 个顶点复测与
1 个线性度对照点，共 **≤28 个测量点 × 3 帧/点 ⇒ 最多 84 个统计帧**。这里的帧数是
`frames_per_point` 的平均化计数，**不是相机原始帧硬上界**：`sweep_focus` 每点还会按峰值
自适应曝光（丢弃帧、`discard=2`、曝光重设后重新采集，峰值落在 20%–70% 满量程即接受），
并在线性度对照点切换曝光档位，因此原始帧数还会增加，实际以 `focus_calibration.json`
的点数及各点 `n_discarded` 为准，事后回写本节。单次会话墙钟按最坏情况预留 **≤30 min**。

### 10.7 上位关系与回滚

- S0‑R1 成功后，**新目录**（`s0/s0r1_dcal_20261124/<UTC戳>/`）的 `optical_path_record.json` 成为
  S0 的**正式记录**；02:42Z 那次**保持原样不改**，作为「探针通过但 D 缺失」的历史证据。
- S0‑R1 失败（D0–D5 任一不过）⇒ S0 仍未闭合，S1 继续封锁，按失败原因另起 R2（不外推、不降级判据）。
- 本节不改变 §2/§6 的判据；若需改判据本身，按 §8 另起带日期的新文件。

---

## 11. R1 实测回写与修订 R2（2026-11-24）：只扩大曝光/SNR 窗口

### 11.1 R1 实测结果（已执行，非推测）

| 项 | 实测 |
|---|---|
| 执行 | `2026-11-24 11:45:02`–`11:46:53`（本地）；SLM #1 + Daheng CCD，全程一次 open/close，无激光操作 |
| 证据目录 | `data/experiments/attribution_20261123/s0/s0r1_dcal_20261124/20261124_034502Z/` |
| 设备 | `drive_board_id` 正常、`readsu_status_at_open=0`、Mono12/满量程 4095；`released_at=11:46:53`、`closed_normally=true` |
| S0 判定 | `probe_status=low_snr`（平场峰值 39–41 counts，噪声底 20.9；离焦响应不可分辨） |
| D 标定 | `status=no_usable_point`：13 个粗扫点在 `0.02–0.2 ms` 内全部 `dim`（clipped 0 / dim 13），未排序、无外扩 |
| D 结果 | **`fresnel_compensation_m_inv=null`**，具体失败原因已写入标准记录；本段不产生任何 D |
| 代码版本 | `aa7c5234`（探针实现与本节预注册） |

与 10:42 那次 S0（同为 `0.05 ms`）对比：平场峰值由约 2803 counts 降到约 41 counts，但全帧
总和仅由 76.27 M 降到 75.25 M。R1 后、R2 前曾据此提出「亮斑被摊开/离焦、曝光窗口不够」的
**初步假设**；§11.3 的 R2 阶梯与标定结果已否决「单纯扩大曝光即可补救」这一假设。峰值下降的
根因仍未定位，不能据全帧总和断言激光功率、离焦或衰减状态。

### 11.2 R2 变更（仅曝光/SNR，判据不动）

- S0 阶梯曝光：`0.05,0.2,1,2 ms`（原 `0.05`）。任一档饱和即按 §2 通用规则降曝，不裁数据。
- 焦面标定自适应曝光窗口：`0.02–2 ms`（原 `0.02–0.2`）。按 `0.05 ms` 实测的 `peak/ms`，
  `2 ms` 足以把峰值抬进可用带；若亮斑随后变亮，自适应曝光会下调，不会过曝。
- **不变**：D 范围 `±1.5 m⁻¹`、`max_extends=1`、13 粗扫 + 7 细化、线性度 `0.8–1.25`、
  光路受阻判据、D 写入条件、`fresnel_compensation_m_inv` 失败保持 `null`；不引用任何历史 D。
- 输出仍为新目录（`s0r2_dcal_20261124/<UTC戳>/`）；R1 目录保持原样，作为失败证据。
- 若 R2 仍 `no_usable_point` 或 `path_blocked` ⇒ **停止软件侧重试**，S1 继续封锁，转为现场检查
  激光/准直与焦面（不靠继续调曝光或放宽判据）。

```powershell
.venv/Scripts/python.exe scripts/slm/probe_optical_path_validity.py `
  --bit-depth 12 --wait-until-idle 600 --exposures "0.05,0.2,1,2" `
  --session s0r2_dcal_20261124 `
  --out-root data/experiments/attribution_20261123/s0 `
  --focus-calibration `
  --focus-floor-ms 0.02 --focus-ceiling-ms 2
```

帧预算与 §10.6 相同（≤28 点 / ≤84 统计帧）；曝光窗口变宽只增加每点的自适应试采帧，
实际仍以 `focus_calibration.json` 的各点 `n_discarded` 与点数回写。

### 11.3 R2 实测结果（已执行，非推测）

| 项 | 实测 |
|---|---|
| 执行 | `2026-11-24 11:56:38`–`11:58:32`（本地）；SLM #1 + Daheng CCD，全程一次 open/close，无激光操作 |
| 证据目录 | `data/experiments/attribution_20261123/s0/s0r2_dcal_20261124/20261124_035638Z/` |
| 设备 | `drive_board_id` 正常、`readsu_status_at_open=0`、Mono12/满量程 4095；`released_at=11:58:32`、`closed_normally=true` |
| S0 判定 | `probe_status=low_snr`；噪声底 20.9；离焦响应不可分辨（宽度/峰值均淹没在噪声） |
| D 标定 | `status=no_usable_point`：13 个粗扫点全部 `dim`（clipped 0 / dim 13），均顶到上界 `2 ms`；未排序、无外扩；`n_discarded_total=0` |
| D 结果 | **`fresnel_compensation_m_inv=null`**；`accepted=false`、`best_diopter_m1=null`，失败原因已写入标准记录 `missing` |
| 代码版本 | 探针实现 `aa7c5234`；运行记录 `git_commit=8032b3f6` |
| 证据哈希 | `focus_calibration.json` sha256 `b8de7a82…`；`optical_path_record.json` sha256 `847c5763…` |

**关键观测：曝光阶梯无响应。** 阶梯各档平场峰值（处理后 counts）：

| 曝光 (ms) | 0.05 | 0.2 | 1 | 2 |
|---|---|---|---|---|
| 峰值 | 40 | 43 | 38 | 41 |

曝光从 `0.05` 提到 `2 ms`（40 倍），峰值基本不变（`38–43 counts`）。这**不符合正常曝光响应**，
与 §11.1「亮斑被摊开/离焦、曝光窗口不够」的初步假设矛盾——若只是曝光不足，峰值应随曝光
近似线性抬升。R2 因此**否决了「单纯扩大曝光即可补救」**；峰值下降根因仍未定位。

**结论与后续（按 §11.2 停止规则）：**

- R2 仍为 `no_usable_point` ⇒ **停止软件侧重试**，不再调曝光、不外扩、不放宽判据。
- `fresnel_compensation_m_inv` 保持 `null`，**S1 继续封锁**。
- 转为**现场检查**：激光功率、准直/光路遮挡、焦面位置与 SLM 面板实际显示，定位峰值异常根因后
  再按 §8 另起带日期的新预注册。
- R1（`s0r1_dcal_20261124`）与 R2（`s0r2_dcal_20261124`）目录均保持原样，作为失败证据。

---

## 12. R3 实测回写（2026-11-24，激光恢复后）：D3 曝光线性度失败，根因已定位

### 12.1 触发与执行

用户指令（2026-11-24）：激光此前关闭、现已恢复，要求重新采集、优先采依赖集物理设备数据。
该指令覆盖 §11.2 的「`no_usable_point`/`path_blocked` 即停止软件侧重试」规则；R3 据此重跑。

| 项 | 实测 |
|---|---|
| 执行 | `2026-11-24 13:15:07`–`13:18:42`（本地；UTC `05:15:07`–`05:18:42`）；SLM #1 + Daheng CCD，全程一次 open/close，脚本不操作激光 |
| 证据目录 | `data/experiments/attribution_20261123/s0/s0r3_dcal_20261124/20261124_051507Z/` |
| 设备释放 | `closed_normally=true`、`released_at=2026-11-24T05:18:42+00:00`；Mono12/满量程 4095 |
| 代码版本 | `772d5f8b` |
| 开局提示 | `13:15:10` 有「读取 SLM 当前显示槽失败 (错误码: 1)」，回退用内存槽 117；非致命，不影响后续相位下发 |

### 12.2 S0 判定：激光确已恢复（`ok`）

- `probe_status=ok`、`exit_code=0`；可用曝光窗口 `0.05 ms`（建议曝光 `0.05 ms`，平场峰值 **1727 counts**，
  噪声底 **20.9 counts**，未饱和）。
- 离焦响应显著：宽度最大变化 **2.95 px（20.8%）**、峰值最大变化 **362.5 counts（21.6%）**，均出现在离焦 `+1 rad`。
- 与 R1/R2（同 `0.05 ms` 峰值仅 `38–43 counts`、阶梯无响应）对比，证实 R1/R2 根因为**激光关闭**，
  非算法/曝光问题。

### 12.3 D 标定结果：`linearity_failed`，D 保持 `null`

- `focus_calibration.json`：`status=linearity_failed`、`accepted=false`、`best_diopter_m1=-0.250`、
  `best_focal_mm=-4000`、`linearity_ratio=1.4527`（>1.25，超 `0.8–1.25`）、18 个扫描点全部 `ok`、
  抛物线顶点 `-0.157`、四点复测后取 `-0.250`。
- 判据 **D3（曝光线性度）未过**，按 §10.4/§10.5 不写 D；`optical_path_record.json` 的
  `fresnel_compensation_m_inv` 保持 `null`，reason 已写入标准记录。**`-0.250` 不予采纳、不外推。**
- 证据 sha256（前 8 位）：`focus_calibration.json=154497d0`、`optical_path_record.json=77d2256f`、
  `optical_path_probe.json=85f9c3fc`。

### 12.4 根因（零设备离线定位，已验证）

`robust_peak(frame) = despike(3×3).max()`（`exposure_adapt.py`），**不做背景/暗场扣除**。
于是峰值含一个加性背景 `B`（环境/杂散光 + 相机偏置）。设 `peak(ms)=k·ms+B`
（`k` = 最优 diopter 处激光信号/ms），则 `peak/ms = k + B/ms` **随曝光变化**，曝光线性度检查
（在 best 曝光 ×0.5 复核）为：

```
linearity = (k + B/0.025)/(k + B/0.05) = (k+40B)/(k+20B)
```

用 best 点（`0.05 ms → 1672.7`）与实测 `linearity=1.4527` 反解得 **`B≈757 counts`、`k≈18307/ms`，
即 `k/B≈24.2`**；据此预测 `0.0935 ms` 峰值 `≈2469`，与细化点实测 `2600–2722` 一致，模型成立。

**关键结论：`k/B` 与曝光无关 ⇒ 只改曝光阶梯无法改变线性度，R4 若仅调曝光必然再次失败。**
要在 ×0.5 复核下过 D3（`linearity≤1.25`）需 `k/B ≥ 60`（当前 24.2），只能靠**提升信背比**或
**软件扣除背景**实现。

### 12.5 R4 处置方向（用户已定夺 2026-11-24）：单曝光扫频 + 修订 D3

候选方案与用户裁决：

| 方案 | 说明 | 裁决 |
|---|---|---|
| A | 软件逐帧背景扣除，使 `peak/ms` 恢复曝光不变性 ⇒ 线性度→1 | 未采纳（改动 `robust_peak` 影响 S0/`adapt_exposure` 全局路径） |
| B | 现场物理提升信背比（压 `B` 或抬 `k`，使 `k/B≥60`） | 未采纳（需现场动光路，本窗口不可控） |
| **C** | **单曝光扫频**：全部点固定同一曝光、按原始峰值排序；D3 由「跨曝光线性度」改为「同曝光重复性」 | **采纳（推荐项）** |

采纳 C 的依据（R3 证据，非推测）：18 点中 `0.05 ms` 粗扫的 `diopter→peak` 曲线干净、两侧单调、
无削顶（`-1.5→1355.7`、`-0.75→1591.3`、`-0.25→1672.7`、`0.0→1669.3`、`+0.25→1522.0`、`+1.25→1127.7`，
满量程 4095）；污染只来自 7 个细化点被暖启动带到 `0.0935 ms`（含端点 `+1.5→1673.3`，其 `peak/ms`
高于所有粗扫点）。**混档才是排序失效的原因，跨曝光归一化则是错误的中间层。**

C 的语义边界（**判据不放松**，只换等价守卫）：

- D0 光路未受阻、D1 `best` 存在、D2 两侧各降 2 点单调、D4 范围 `±1.5 m⁻¹`/`max_extends=1`、
  D5 证据可复算 —— 全部不动。
- D3 在单曝光下**不可能**做「×0.5 曝光复核」（`B` 是加性常数，任何跨曝光比值都天然偏 1），
  故 D3 改为**同曝光重复性**：重显最优点相位、用同一曝光重测 `frames` 帧，
  `repeat_ratio = alt_peak / best_peak` 必须落在 `[0.90, 1.111]`；越界 ⇒ `repeatability_failed`，
  缺失 ⇒ `repeatability_unavailable`。单曝光下 `peak_per_ms` 与原始 `peak` 单调等价（分母为常数）。
- 削顶保护：任一 pass 出现 `fraction > ACCEPT_HIGH=0.80` 时整段收缩曝光
  （`exp *= TARGET_FRACTION/max_fraction`，夹到 `[floor, ceiling]`）后**全网格重测**（≤3 pass），
  保证排序始终在**同一个未削顶曝光**下成立。

已知残余不确定性（必须与 D 一起写进回写，勿当精确值）：
`0.05 ms` 下 `-0.25` 仅比 `0` 高 3.4 counts（噪声 ~12），峰顶平坦 ⇒ 顶点本身有 `±0.25 m⁻¹` 量级不确定性，
故 R4 把预注册曝光抬到 `0.1 ms`（信号翻倍、`B` 不变，best 峰值约 `0.63` 满量程，仍在 `0.80` 门限内）。

### 12.6 R4 预注册（执行前锁定的参数与命令）

代码侧（已落地）：`focus_probe.py` 增 `EXPOSURE_MODE_ADAPT/FIXED`、`FIXED_EXPOSURE_RETRIES=2`、
`FIXED_REPEAT_MIN=0.90`/`FIXED_REPEAT_MAX=1.111`、`_fp_same_exposure_repeat`、`_fp_fill_coarse_grid`；
`FocusProbeResult` 增 `exposure_mode`/`fixed_exposure_ms`/`repeat_ratio`；
`probe_optical_path_validity.py` 增 `--focus-exposure-mode {adapt,fixed}`、`--focus-fixed-exposure-ms`，
`_evaluate_focus_result` 按模式分支。adapt 路径逐位不变（既有 81 项测试全绿）。

| 参数 | 值 | 说明 |
|---|---|---|
| `--focus-exposure-mode` | `fixed` | 单曝光扫频 |
| `--focus-fixed-exposure-ms` | `0.1` | 预注册唯一曝光（覆盖 S0 推荐 `0.05`；越界按 floor/ceiling 夹取） |
| `--focus-coarse-points` / `--focus-diopter-range` | `13` / `1.5` | 与 R1–R3 一致 |
| `--focus-refine-points` / `--focus-frames` | `7` / `3` | 与 R1–R3 一致 |
| `--focus-floor-ms` / `--focus-ceiling-ms` | `0.02` / `2` | 与 R2–R3 一致 |
| 焦面中心 / 半径 | `960,600` / `450`（panel px） | 冻结值，与 S0 一致 |
| 判据 | D0–D5 同 §10.4；D3 → `repeat_ratio ∈ [0.90,1.111]` | 不新增、不放宽任何判据 |

```powershell
.venv/Scripts/python.exe scripts/slm/probe_optical_path_validity.py `
  --bit-depth 12 --wait-until-idle 600 --exposures 0.05 `
  --session s0r4_dcal_20261124 `
  --out-root data/experiments/attribution_20261123/s0 `
  --focus-calibration --focus-exposure-mode fixed --focus-fixed-exposure-ms 0.1 `
  --focus-centre-px 960,600 --focus-radius-px 450 `
  --focus-coarse-points 13 --focus-diopter-range 1.5 `
  --focus-refine-points 7 --focus-frames 3 `
  --focus-floor-ms 0.02 --focus-ceiling-ms 2
```

回滚/失败处置：R4 若仍 `repeatability_failed`/`repeatability_unavailable`/`path_blocked`/`out_of_range`
⇒ D 保持 `null`、S1 继续封锁，转为现场物理路径（方案 B）；**不得**改用新的曝光阶梯、不得放宽
`[0.90,1.111]`、不得引用历史 D 值。

### 12.7 R4 实测回写（2026-11-24）：单曝光模式在线实现、D2 单峰失败 ⇒ D 撤回

| 项 | 实测 |
|---|---|
| 执行 | `2026-11-24 18:27:01`–`18:29:40`（本地；UTC `10:27:01`–`10:29:40`）；SLM #1 + Daheng CCD，一次 open/close，脚本不操作激光；SLM 控制器在开跑前由用户重新上电/插拔（脚本内 `power_cycled_this_session=false`） |
| 证据目录 | `data/experiments/attribution_20261123/s0/s0r4_dcal_20261124/20261124_102701Z/` |
| 释放 | `occupancy.closed_normally=true`、`released_at=2026-11-24T10:29:40+00:00`、Mono12/4095 |
| 代码版本 | `64318de3`（R4 代码未提交，工作树改动见 §12.6） |
| sha256(前 8) | `focus_calibration.json=f1ce35cf`、`optical_path_record.json=ee3ed2df`、`optical_path_probe.json=2e1e8d0b` |

- **R4 的曝光/重复性设计生效**：18 点全部 `exposure_ms=0.1000`（无混档、无削顶 `peak_fraction=0.36–0.59`），全部 `status=ok`，`repeat_ratio=0.949`（在 `[0.90,1.111]` 内），`linearity_ratio=null`（fixed 模式按设计不再做跨曝光线性度）。S0 `verdict.state=ok`、`exit_code=0`，离焦宽度变化 `3.27 px (21.0%)`、峰值变化 `263.9 counts (17.9%)`。
- **D2（单峰）失败，D 撤回**：`best_diopter_m1=-0.5833` 的左侧 `peak/ms` 为
  `-0.750→23477, -0.667→23227, 最优→24327`，右侧 `-0.500→24093, -0.417→24210`，
  两侧都没有「各单调下降 2 点」。峰顶 `[−0.75,−0.25]` 起伏仅约 `4%`（与 3 帧复测噪声
  `σ_peak≈121.5 counts` 同量级），而 `450 px / ±1.5 m⁻¹` 的探头只折算约 `0.7` 波前，峰顶天生平坦。
- **evaluator 缺陷已修**：`_evaluate_focus_result` 原来漏掉 D2，导致 R4 当次把 `-0.5833` 写进了
  `optical_path_record.json`。现已补 D2 单峰门禁（`ambiguous` ⇒ 不写 D），并把真实 R4 证据只读重放，
  得到 `("ambiguous", "最优点 -0.583 m⁻¹ 两侧 peak/ms 未各单调下降 2 点 …")`。
  **该记录里的 `-0.5833` 作废，任何下游不得引用**（含 sha256 `ee3ed2df` 那份）。
- **D5 一致性缺陷已修**：原 `notes` 只记抛物线阶段「四点复测后取 −0.500」，精扫换赢家后未补记；
  现在追加「精扫后最优点改为 −0.583 m⁻¹」，证据自洽。
- 门禁：两个测试文件 `85 passed`、`ruff 0`、无 BOM；实测证据目录未被改写（`data/` 无新提交改动）。

### 12.8 R4b 预注册（实验前锁定；只提信噪比，不改判据/网格）

R4 证明扫描控制和单曝光重复性没问题，剩下唯一的失败点是 D2 处的峰顶起伏与噪声同量级。
R4b 只把每点统计帧数 `3→12`（独立帧平均，`σ` 理论降 `√4=2` 倍），**其余所有参数与判据保持
§12.6 冻结值**（含粗扫 13 点、范围 `±1.5`、细化 7 点、固定曝光 `0.1 ms`、`max_extends=1`、
`repeat_ratio∈[0.90,1.111]`、D0–D5 全部不动、D2 严格单调不动）。

```powershell
.venv/Scripts/python.exe scripts/slm/probe_optical_path_validity.py `
  --bit-depth 12 --wait-until-idle 600 --exposures 0.05 `
  --session s0r4b_dcal_20261124 `
  --out-root data/experiments/attribution_20261123/s0 `
  --focus-calibration --focus-exposure-mode fixed --focus-fixed-exposure-ms 0.1 `
  --focus-centre-px 960,600 --focus-radius-px 450 `
  --focus-coarse-points 13 --focus-diopter-range 1.5 `
  --focus-refine-points 7 --focus-frames 12 `
  --focus-floor-ms 0.02 --focus-ceiling-ms 2
```

- 帧预算：`≤18 点 × 12 帧 + 复测`，约 220 帧；仍为独立点，总计 ≤250 帧（相对 R4 的 3 帧 ×4）。
- 成功 ⇒ D 按 §10.5 写入 S0 标准记录并解锁 S1；失败（含再次 `ambiguous`）⇒ D 继续 `null`、
  S1 继续封锁，**不得**放宽 D2 或改判据；此时必须把「峰顶在 450 px/±1.5 m⁻¹ 下物理平坦」作为
  结论交用户定夺是否另起探头几何/度量修订的 R5（用户签字前不做）。

### 12.9 R4b 实测回写（2026-11-24）：帧数 ×4 仍 `ambiguous`，D 保持 `null`

| 项 | 实测 |
|---|---|
| 执行 | `2026-11-24 18:48:52`–`18:51:28`（本地；UTC `10:48:52`–`10:51:28`）；SLM #1 + Daheng CCD，一次 open/close，脚本不操作激光 |
| 证据目录 | `data/experiments/attribution_20261123/s0/s0r4b_dcal_20261124/20261124_104852Z/` |
| 释放 | `closed_normally=true`、`released_at=2026-11-24T10:51:28+00:00` |
| sha256(前 8) | `focus_calibration.json=773358c0`、`optical_path_record.json=655bec41` |
| 参数核对 | 18 点全为 `exposure_ms=0.1`、`frames=12`，`repeat_ratio=0.977`（优于 R4 的 0.949），无削顶，S0 `verdict.state=ok` |

- 结果：`status=ambiguous`、`accepted=false`、`best_diopter_m1=-0.500`，
  原因「最优点 -0.500 m⁻¹ 两侧 peak/ms 未各单调下降 2 点 (左: -0.667→19912, -0.583→19748, 最优→20605)」。
  `optical_path.fresnel_compensation_m_inv=null` —— D2 门禁正确拦下，未写 D。
- 峰顶 `peak/ms`：`-0.583→19748, -0.500→20605, -0.417→20084, -0.250→20596, 0→20398`。
  **帧数 ×4 后仍不是单峰**，`-0.50` 与 `-0.25` 基本等高（差 9 counts/ms ≈ 0.04%），
  而 `-0.583` 反而低了 857 counts/ms（4.2%）。说明这不是单纯的随机帧噪声：
  在 `450 px / ±1.5 m⁻¹` 的探头几何下，峰顶本身平坦且多点局部极大，D2「严格单调 2 点」
  对该物理曲线不可满足。**继续堆帧数/重扫同一配置没有意义。**
- 按 §12.8 的预先承诺：D 保持 `null`、S1 继续封锁；D2 与全部判据不放宽，
  是否改探头几何/度量（R5）交用户定夺。

### 12.10 用户定夺（2026-11-24）：显式风险接受近似 D ≈ −0.5 ± 0.25 m⁻¹，解锁 S1

用户在 R4b 结果页明确选择「接受 D≈−0.5±0.25 并解锁 S1」。这是**有意识的风险接受**，不是
D2 通过：R4/R4b 两次单曝光扫描都给出同一区间（R4 best `−0.583`、R4b best `−0.500`，
峰顶覆盖 `−0.75…−0.25`），但 D2 严格单峰两次都失败，所以**不存在精确标定的 D**。

- **写入约定**：下游 S1 使用 `D_accepted = −0.5 m⁻¹`（`focal_mm = 1000/D = −2000 mm`，
  与 `focus_probe` / `content_axis_2x2_capture` 共用 `1000/D` 不翻符号的约定），
  不确定度 `±0.25 m⁻¹` 必须与结果一起记录；`optical_path_record.json` 中的原始
  `fresnel_compensation_m_inv` 仍保持 `null`（不改原始证据），近似值单独记录在本节
  与 S1 运行的 provenance 中。
- **风险量级（可复算）**：SLM pitch `2.2 µm`、半径 `450 px` ⇒  pupil 半径 `r ≈ 0.99 mm`；
  若真实 D 偏离 `0.25 m⁻¹`，面板边缘残余相位 `π D r² / λ ≈ 0.72 rad ≈ 0.115 λ`
  （`λ=1064 nm`）。对本次 4 帧内容判别实验属小量；**不得外推**为 D 已精确标定。
- **禁止事项**：不得引用历史 `+0.559` 或旧 D；不得修改 R4/R4b 原始证据；不得把 D2 从
  「严格单调 2 点」改松；凡 S1 结论对补偿量敏感（轴归属/幅值接近判据边界），必须用
  `D ∈ {−0.75, −0.25}` 做 bracket 复测，否则结论标注「未做 D bracket」。
- **S1 对照要求**：同窗口内至少一对 `D=0` vs `D=−0.5` 的配对采集，并把每帧实际下发的
  Fresnel 相位哈希与曝光写入 provenance；S1b 的 2×2 因子顺序按 §2.1 冻结。
- 门禁代码：`82fc971a`（R4 fixed 模式 + D2 门禁）；R4b 证据
  `data/experiments/attribution_20261123/s0/s0r4b_dcal_20261124/20261124_104852Z/`。


---

### 12.11 S1b 配对实测回写（2026-11-24）：内容无关签名成立但 §2.1 严格判决 `insufficient_consistency`，S1 结论 BLOCKED

**执行范围**：只读判读，无新增设备占用。S1b 采集数据已在本窗口早前落盘
（`data/experiments/attribution_20261123/s1/s1b_content_axis_20261124/`），本节只做离线判读与
结论登记；判读脚本 `scripts/slm/analyze_content_axis_2x2_20261124.py`（只读，不写采集目录、
不碰设备），测试 `tests/ao_shaping/scripts/test_analyze_content_axis_2x2_20261124.py`。

**采集合规证据（两组均满足）**：`d0_20261124`（D=0）与 `dm0p5_20261124`（D=−0.5）两组
`summary.status=ok`、`n_frames=4`、`saturation_count=0`；逐帧曝光 `0.087 ms`（与 S0 标定
`0.1 ms` 不同，勿混用）；`max_grayscale=1023` / gray_modulus `1024`；`released.json` 中
`slm_closed=true`、`camera_closed=true`、`lock_released=true`、`error=null`，`closed_normally=true`；
脚本未操作激光。每帧 `phase_sha256`/`gray_sha256` 已写入 provenance；`row0_step_by_pattern`
记录 B 相对 A 的第 0 行灰度阶跃 `512`（`3.1447 rad ≈ π`，受 1023/1024 量化影响有 ≈0.003 rad
偏差），其余行阶跃为 `0`。D=−0.5 组 `focus_compensation.focus_d_m1=−0.5`、
`focal_mm=−2000`（沿用 §12.10 `1000/D` 不翻符号约定）。

#### 12.11.1 判读口径（冻结，复现 §2.1）

- `principal()` 严格复现 issue247 约定：`ref` = A1 flat 0th `(1105.25, 1227.56)`，
  `exclude_core=6`、`min_dist=15`、`find_spots` min_dist=20 / thresh=`0.01×max`。
- **零级参考不用逐帧 argmax**：内容 B 的第 0 行 π 阶跃会把逐帧 0th 质心扰动约 3 px
  （A128 帧 0th 行 1104 vs B128 1101），若用它作 ref 会虚假放大 B 的幅值；A/B 的
  principal 主瓣实际落在同一绝对 CCD 位置。逐帧 0th 仅作漂移诊断输出。
- **ROI 一律由预测位置导出**（§2.1 L142）：对 `K_col·N`、`K_row·N`、
  `K_col·N/1.6`（L145 压缩律）三个预测位置各开 `7×7` ROI，峰值为判读辅助；
  **未使用** 11-23 那个固定 `core_exclude=34.96 px`（它会排除 y 族 p256 真 ±1）。
- 弱信号标志：主瓣峰值 < `0.10×0th` 峰值记 `weak_signal`，该组不可判轴。

#### 12.11.2 D=0 组实测（SNR 充分，4/4 帧）

| 格 | 内容 | N | 主轴 | 幅值 px | K | principal 峰 | `|amp−K_col·N|` | `|amp−K_row·N|` | `|amp−K_col·N/1.6|` |
|---|---|---|---|---|---|---|---|---|---|
| A128 | A | 15 | **row** | 56.32 | 3.755 | 610 | 39.68 | 36.68 | 3.68 |
| A256 | A | 7.5 | **row** | 27.70 | 3.694 | 683 | 20.30 | 18.80 | 2.30 |
| B128 | B | 15 | **row** | 56.82 | 3.788 | 578 | 39.18 | 36.18 | 3.18 |
| B256 | B | 7.5 | **row** | 28.72 | 3.830 | 670 | 19.28 | 17.78 | 1.28 |

- **A/B 不可区分**：B128 主瓣与 A128 相差 <0.5 px、B256 与 A256 相差 <1.0 px（同一绝对
  CCD 位置）；这是针对「第 0 行 π 阶跃」扰动的真实**内容无关零签名**。
- **绝对轴/幅值与冻结约定矛盾**：四格全部落 **CCD 行轴**（§2.1 预期内容 A 落列轴），
  幅值 `≈3.7–3.8·N`，既非 `K_col·N=6.4·N` 也非 `K_row·N=6.2·N`（残差 17–40 px ≫ 1.5 px），
  即使按 L145 压缩律 `K_col·N/1.6=4.0·N` 也只有 B256 一格残差 ≤1.5 px。
- **预测位置 ROI 为背景**：`K_col·N` 与压缩 `K_col·N/1.6` 的预测列轴 ROI 峰值仅 18–44
  （帧中位 15），`K_row·N` 预测行轴 ROI 峰值 22–102；主瓣（578–683）不在任何预测位置。
  真 ±1 阶落在行轴 `dr≈+56/+28`，而非 93/46.5。
- 0th 未被抑制（最亮 1173–1294；理想 0/π 应≈0），±1st 不对称（A128 +610/−267），表明
  二进制光栅非理想（灰度量/π 误差与倾斜）。

**§2.1 严格判决**：`insufficient_consistency`。理由：内容 A 未落列轴；四格幅值均不服从
`K·N` 律（残差远超 1.5 px）；A/B 同址只是必要非充分签名，按 L135–136 不能据此宣布
`content_independent`。此判决与「内容相关 4/4」明确互斥，**不构成内容相关路径确认，
也不构成干净的内容无关实锤**。

#### 12.11.3 D=−0.5 组实测（弱信号，不可判轴）

四格 principal 峰值 44–59，仅为 0th（约 1200）的 3–4%，`weak_signal=true`；A256 的
principal 甚至落在噪声像素 `dr≈−1100`。位移阶近乎消失，轴归属不可靠。

**§2.1 判决**：`insufficient_signal`（主瓣近噪声，轴不可判，不计入分母）。D=−0.5 组与
D=0 组一样 A/B 不可区分，但因本组无轴信号，不能作为 D 稳健性的佐证。

#### 12.11.4 A1 ↔ S1b 轴/K 矛盾协调（@oracle 复核，2026-11-24）

同一 x 向二进制 0/π 光栅：A1（11-21）`grating_x` → CCD **列轴** `K≈6.35`（p128 约 95 px）；
S1b（11-24）内容 A → CCD **行轴** `K≈3.78`（p128 约 56 px）。两者构造同为二进制 0/π、
同样 `slm.display_data` 下发、同面板形状 (1200,1920)。复核结论：

- **层 1（置信 ~0.8）：A1 前提并非有效真值**。A1 自身判据为 `mapping_still_ambiguous` /
  `frozen=false`：其全幅光栅/tilt 阵营出现「两个面板轴都成到同一 CCD 列轴」这一不可逆
  2×2 映射不可能的结果，已被标为异常；L145 明确把全幅 fy≡0 类排除出 A 映射校验
  （「列轴 + 1920/1200 内容压缩」）。S1b 的 `≈3.78·(1920/p) = ≈6.05·(1200/p)` 恰与
  A1 `grating_y` 尺度（p128 59.27 px）及预注册 K×1200≈7613 相容；S1b 的行轴也与 A1
  唯一自洽的子映射（`ladder_x → 行轴`）一致。换言之冻结主表的 `K_col·N=96/48` 可能嵌入了
  1.6× 归一化错误（1200 采样 bin 间距 × 1920 采样 bin 索引），S1b 幅值才是物理正确尺度。
- **层 2（置信 ~0.6，机制未定）：仍存在真实会话态差异**。同为二进制构造、同为 memory 模式，
  11-21 给出压缩后的列轴斑、11-24 给出正确尺度的行轴斑。候选：光机状态改变（重新装夹/
  旋转）或驱动板/固件存储态（A1 `power_cycled_this_session=null`）；若「压缩」来自缓冲区
  transpose/reshape（1200×1920 ↔ 1920×1200），可同时解释 1.6× 尺度与轴交换，但未证实。
  两会话 2π 灰度确实不同：A1 `two_pi_grayscale=993`
  （`data/debug/hw_window_20261121/issue247_axis/optical_path_record.json:13`），S1b
  `max_grayscale=1023` / gray_modulus `1024`（见 12.11 采集证据）。
- **已排除**：blaze-vs-binary 路径差异（A1 光栅同为二进制，`issue247_geometry_capture_20261120.py:328-341`）；
  纯缩放单独导致轴交换（缩放不改轴）；内容类专属上传路径（两会话同类同 API）。

#### 12.11.5 S1 结论与最小解歧实验

**S1 结论状态：BLOCKED（未闭合）**。映射/标定矛盾未协调前，不确认内容相关路径，也不宣布
干净的内容无关实锤；按 §2.1 登记 `insufficient_consistency` + 「须做映射/标定协调」。
G1（A 映射闭合）仍缺 `blaze2d` 定 2×2 的实测，不能据 S1b 反推冻结映射。

**最小解歧实验（同一次占用，约 6–8 帧，与 S1b 同一光路态）**：

1. `grating_x_p128`（用 A1 builder `issue247_geometry_capture_20261120.py:328` 构造，走 S1b
   下发路径），同时给出会话内 flat 0th 与 A1 冻结 ref；可加一组相同 gray 数组在
   `2π=993` 与 `2π→1024` 两种灰度下的对照。
2. `grating_y_p128`（对照，A1 中已是正确尺度）。
3. `blaze2d`（如 px128/py192，`sx=+1, sy=+1`）——唯一能钉死 2×2 映射的探针族
   （A1 README 自述）。
4. `ladder_x`（复现唯一自洽的 panel-x → 行轴）。

判据：若本会话 A1-builder `grating_x` → 行轴约 56–59 px，则 A1 列轴结果为会话态，弃用
A1 x 族作参考；若它 → 列轴 95 px 而同会话 S1b 内容 A → 行轴 56 px，则隔离 gray 数组/
上传路径（比对 phase/gray SHA256、panel_res、2π 尺度）。`blaze2d` 无论哪支都给出映射定论。

#### 12.11.6 D bracket 状态与 D 原始值

按 §12.10「凡 S1 结论对补偿量敏感（轴归属/幅值接近判据边界）须用 `D∈{−0.75,−0.25}`
bracket 复测，否则标注『未做 D bracket』」：本 S1b 在 D=0（SNR 充分）已因幅值违反 `K·N`
律而失败，D=−0.5 为弱信号失败测量而非 D 稳健性证据；补 D 点救不回幅值门，只在映射协调后
作为轴随 D 稳定性的次级检查。**本次标注：未做 D bracket，bracket 复测降级到解歧实验之后**，
并把弱信号帧预声明为不计入分母。

`optical_path_record.json` 的原始 `fresnel_compensation_m_inv` 仍保持 `null`；近似值
`D_accepted=−0.5±0.25 m⁻¹` 仅存在于 §12.10 与 S1 provenance；未引用历史 `+0.559` 或旧 D，
未修改 R4/R4b 原始证据，未放宽 D2。

#### 12.11.7 门禁与代码

- 判读脚本：`scripts/slm/analyze_content_axis_2x2_20261124.py`（RO 判读）；
  测试：`tests/ao_shaping/scripts/test_analyze_content_axis_2x2_20261124.py`（5 passed，
  `--import-mode=importlib`）。
- 门禁：`ruff check` 两文件 `All checks passed`；判读脚本无 UTF-8 BOM；`py_compile` 通过。
- 证据：`data/experiments/attribution_20261123/s1/s1b_content_axis_20261124/{d0_20261124,dm0p5_20261124}/`。
- D 标定实现门禁（早前）：`aa7c5234`；R4/R4b 门禁 `82fc971a`；§12.10 决策 `1a602f87`。
