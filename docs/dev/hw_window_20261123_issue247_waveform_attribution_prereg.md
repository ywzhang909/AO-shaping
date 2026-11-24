# #247 波形族归因窗口预注册 v2（补齐 ± / 匹配空间频率 / 独立验证）· 2026-11-23

> **状态**：用户已明确选择「重新开窗补齐」，本文件在**下发任何图样之前**写定，是该窗口
> 唯一判据来源。采集后不回改判据、不回改图样表、不按结果切换优先级标签。
> 上游：`docs/dev/issue247_waveform_diagnostic_plan_20261121.md`；
> 今日已完成窗口：`docs/dev/hw_window_20261123_issue247_waveform_prereg.md`（数据
> `data/debug/hw_window_20261123_issue247_waveform_probe/`，verdict `patch_position_specific`）。
>
> **本窗口与今日窗口的关系**：今日窗口已给出 `waveform_gate` pass（grating 与连续 blaze 同轴、
> 差 0.32°）、`axis_gate` fail（panel-x/y 都落 CCD 列）、`patch_gate` fail（局部 patch 与 1-D 族差
> 89°）；但只有连续 blaze 有 ±，grating/tilt/patch 三族没有齐备的 ± 对照，也没有预注册
> held-out 方向。本窗口专门补这两项，**不改变**「映射仍歧义、不冻结」的总状态。

---

## 1. 窗口身份

| 项 | 值 |
|---|---|
| 设备 | Santec SLM #1（1064 nm）+ Daheng MER2-507-23GM cam 0 + 1064 nm 激光 |
| 采集脚本 | `scripts/slm/issue247_geometry_capture_20261120.py --plan attribution --execute` |
| 分析脚本 | `scripts/slm/issue247_waveform_probe_solve_20261122.py`（若需扩展，先离线自测再判读） |
| 输出目录（全新，已确认不存在） | `data/debug/hw_window_20261123_issue247_waveform_attribution/` |
| 前置门 | `probe_optical_path_validity.py --session issue247_attribution_20261123 --wait-until-idle 1800`，verdict 必须 `ok`，且 0.04 ms 落在其可用曝光区间内 |
| 曝光 | **0.04 ms**（相对默认 0.05 ms 的偏离理由：今日同功率 0.05 ms 曾越饱和门，0.04 ms 实测峰值 2199 counts、外推最坏 2954 < 门限 3685.5；采集前冻结于此，若前置门不支持则停手请用户裁决） |
| settle / 帧数 / 饱和门 / 位深 | 0.8 s / 4 帧每图样 / 0.9×满量程 / Mono12 全幅 |
| open/close | 单次 `--execute` 内恰好一次 open + 一次 close，覆盖全部 23 图样 |

## 2. 图样表（冻结；顺序固定；23 图样 × 4 帧 = 92 帧）

### 2.1 判据集（fit set）：同轴匹配空间频率 128 px，三族 ±

| # | ID | 族 | 面板轴 | sign | 说明 |
|---:|---|---|---|---|---|
| 1 | `flat_pre` | flat | — | — | 前参考 |
| 2 | `grating_x_p128` | binary grating | x | ±（同时） | 0/π，周期 128 px；对称 ±1 级同帧出现 |
| 3 | `tilt_x_plus` | continuous tilt | x | + | 连续 ramp，周期 128 px |
| 4 | `tilt_x_minus` | continuous tilt | x | − | 同 3，反号 |
| 5 | `patch_x_plus_p128` | localized patch | x | + | 600×600 局部 ramp，周期 128 px |
| 6 | `patch_x_minus_p128` | localized patch | x | − | 同 5，反号 |
| 7 | `grating_y_p128` | binary grating | y | ±（同时） | 0/π，周期 128 px |
| 8 | `tilt_y_plus` | continuous tilt | y | + | 连续 ramp，周期 128 px |
| 9 | `tilt_y_minus` | continuous tilt | y | − | 同 8，反号 |
| 10 | `patch_y_plus_p128` | localized patch | y | + | 600×600 局部 ramp，周期 128 px |
| 11 | `patch_y_minus_p128` | localized patch | y | − | 同 10，反号 |
| 12 | `flat_mid` | flat | — | — | 漂移参考 |

**二值 grating 的 ± 口径（写死，避免事后挑峰）**：二值 0/π 光栅的幅度传递函数为实函数，
±1 级强度恒等、镜像图样与自身等价，因此不做单独 `*_minus` 图样；方向向量定义为
**−1 级质心 → +1 级质心**（不是 0 级 → 某一级），两个峰都必须在报告里出现。若分析器只报
单峰，则其输出对 grating 不算有效测量，须离线补双峰后再判，不得用单峰角度代替。

### 2.2 held-out 集（独立验证，绝不参与 §3 任何归因/门判定）：周期 192 px

| # | ID | 族 | 面板轴 | sign |
|---:|---|---|---|---|
| 13 | `grating_x_p192` | binary grating | x | ±（同时） |
| 14 | `tilt_x_plus_p192` | continuous tilt | x | + |
| 15 | `tilt_x_minus_p192` | continuous tilt | x | − |
| 16 | `patch_x_plus_p192` | localized patch | x | + |
| 17 | `patch_x_minus_p192` | localized patch | x | − |
| 18 | `grating_y_p192` | binary grating | y | ±（同时） |
| 19 | `tilt_y_plus_p192` | continuous tilt | y | + |
| 20 | `tilt_y_minus_p192` | continuous tilt | y | − |
| 21 | `patch_y_plus_p192` | localized patch | y | + |
| 22 | `patch_y_minus_p192` | localized patch | y | − |
| 23 | `flat_post` | flat | — | — |

held-out 与 fit 集在**空间频率（192 vs 128 px）**上完全不重叠；在归因结论冻结前，任何人
（含主 agent）不得查看 held-out 图样的 dr/dc/角度、不得用它调参数、不得拿它挑判据。
判读脚本须物理隔离 fit 集结果文件后才生成 held-out 指标。

## 3. 预注册判据（采前冻结）

### 3.1 数据有效性门（先判，不过则整窗 invalid）
- `summary.status == ok`、`saturation_count == 0`、`n_frames == 92`、23 个 ID 全到齐。
- flat 漂移：`flat_pre / flat_mid / flat_post` 两两质心差最大值 ≤ 2.0 px（沿用今日口径）。

### 3.2 归因门（**只用 §2.1 判据集**，全部逐项报告，禁止单一优先级标签覆盖其他 fail）
角 θ(f,a) 定义：0 级 → ±1 级的位移向量在 CCD 平面上的角度（grating 用 −1→+1 向量）。
- **G1 波形依赖门**：固定面板轴 a，三族角度极差 > 5° ⇒ 偏转方向随**波形**变化 ⇒ 归因
  「灰度/LUT/载频机制」。
- **G2 单族异常门**：恰有一族 f 在 x 与 y 两轴上均与另两族中位数差 > 5°，而另两族互相
  ≤ 2° ⇒ 归因「该族生成或下发路径」。
- **G3 轴分离门**：每族分别算 |θ(f,x) − θ(f,y)|，期望 88.544°（A 映射口径）、容差 2°；
  若某族该值 ≈0° 记为该族**轴路由失败**。
- **G4 符号一致性门**：每族每轴 `θ(+) − θ(−)` 期望 180°，容差 5°。
- **G1/G2 同时 pass 时的唯一归因口径（写死）**：若恰有一族异常（G2 pass），则以 G2 为准，
  归因「该族生成或下发路径」，G1 的 pass 只作为「波形间确有方向差」的描述，**不得**
  改判为灰度/LUT/载频机制；只有 G2 fail 而 G1 pass，才归因灰度/LUT/载频机制。
- verdict 为 `{G1,G2,G3,G4}` 的 pass/fail 集合 + 每个门的实测数值，并给出
  `attribution_label`（按上一行优先级得出）；今日 `patch_position_specific` 那种
  「优先级标签盖住其他 fail」的写法**禁止复用**——本窗口的 label 必须与逐门值同时呈现。
- 本 plan 不含 `blaze_*` 前缀图样，分析器旧的 6 值 `verdict` 中依赖 blaze 的门会 skipped，
  因此**旧 `verdict` 对本窗口不适用**；权威结论只看 `attribution_gates` 与
  `held_out_validation`（报告里须写明这一点，禁止把旧 verdict 当结论引用）。

### 3.3 独立验证（在 §3.2 完全冻结后才允许执行）
- 角度口径统一用分析器的 `axis_angle_deg`（二值 grating 为 ±1 双峰模 180°合并后的轴向角，
  **不是** `principal.angle_deg`），定义 CCD 列 ≈0° / CCD 行 ≈90°。
- 预测规则由 §3.2 唯一决定，冻结后不回改：
  - 若 G1 pass（波形依赖）：held-out 预测 `θ̂(f,a) = θ_fit(f,a)`（各族保留自己的轴角）；
  - 若 G2 pass（单族异常 f*）：异常族 `θ̂(f*,a)=θ_fit(f*,a)`，其余族
    `θ̂(f,a)=median_g≠f* θ_fit(g,a)`；
  - 若 G1/G2 都 fail 且 G3 全过（轴路由一致）：`θ̂(f,a)=median_g θ_fit(g,a)`；
  - 否则 `θ̂` 记为 `not_assessable`，held-out 门判 fail，不写外推结论。
- 轴向类别：`θ mod 180` 离 0° 更近记 `column`，离 90° 更近记 `row`；实测类别同样从
  held-out 的 `axis_angle_deg` 取。符号类别用 `lobes[]` 的 side 与 0 级质心相对方向核对。
- 指标：`held_out_axis_accuracy = 预测轴向正确的图样数 / 10`（不含 flat_post），并逐族报；
  符号一致性另报 `held_out_sign_accuracy`。
- **只有 held_out_axis_accuracy ≥ 0.9 且每族 ≥ 2/3，才允许写「归因规则可外推到未见方向」**；
  即便如此，本窗口仍**不宣称「矫正提升预测准确率」**（那是拟合矫正系数后的另一实验），
  也不改变 #247「映射仍歧义、不冻结」。未过则不写任何外推结论。

## 4. 设备纪律
- 开始前/结束后在回复中明确报告 SLM #1 / CCD / 激光的占用、动作、释放时刻、是否正常关闭。
- 前置探测与采集不同时运行；均带 `--wait-until-idle`；USB 驱动板假死（ReadSU=-1000 / -10002）
  当日停手，不自动 reopen/retry。
- 采集后核验 `released.json`：`slm_closed / camera_closed / lock_released / closed_normally` 全 true，
  且无残留 bench Python 进程。
- SLM 关闭前先 `set_grayscale(0)`（由采集脚本保证）。

## 5. 明确不做
- 不 push、不改 GitHub issue 状态、不改映射冻结状态、不写正式 `issue247_axis_map.json`。
- 不发布 LUT / 效率标定 / #238 g₂π，不声称 #234/#236 有增益。
- 不使用 held-out 调参，不在采后回改本文件任何判据或图样表。
