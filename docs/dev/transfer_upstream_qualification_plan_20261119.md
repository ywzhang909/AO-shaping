# Transfer 上游资格补齐与再采计划（2026-11-19）

> **状态（2026-11-20 更新）：P0/P1/P2/P3 全部完成；P4 双门已用新采数据判定，均为有效 `fail`。
> 不再有"设备窗口未开"的阻塞——P2（同曝光暗场）与 P3（train/holdout 拆分数据）已在
> 2026-11-20 单一串行窗口采集完成并正常释放，P4(a) refit 与 P4(b)(c) 双门评估已完成：
> 强度门 12/12 可评估、最差 `relL2=0.8849` ≫ δ=0.20（`verdict=fail`）；方向门 pooled 4/14 < 10
> （`verdict=fail`，pass1 2/7、pass2 2/7）。证据 `data/debug/slm_transfer/p3_gate_20261120/`。
> 结论：transfer `light/none` 科学验收继续暂停，不进 P5；不得调阈值或换模型"抢救" fail
> （离线调查见 `agent_notes_20261120.md` §10）。历史 2026-11-19 状态：P0 完成（评估器 + runner + 15 单测）、
> P1 完成（archived 候选全部 `not assessable`，暗场与曝光不匹配）、P2/P3 规格按 §6 冻结、
> 方向门 P3a 采集脚本与评估器离线回归通过（复现归档 9/14 / exit 1）。**
> 触发：2026-11-19 用户在三岔口中选择「补上游再采（推荐）」——先补齐模型资格与测量口径，
> 再做 transfer `light/none` 采集。
> 关联：`transfer_strength_gate_preregistration_20261119.md`（δ=0.20 与门定义，
> 本文件不修改 δ）、`hw_ccd_mode_test_plan_20261118.md`（验收目标）、
> `agent_notes_20261119.md` §Transfer 模型资格复核。

---

## 0. 一句话结论

**「锁前置」当前无法完成，且不是换 checkpoint 能解决的。** 现有归档数据里
没有任何 checkpoint 同时通过方向门与强度门；强度门所要求的评估器本轮才实现（P0）。
归档 probe 是 **Mono8 8-bit**（`_optical_probe.json: bit_depth=8`）；暗场库里虽已有
8-bit（0.024/0.043/0.047 ms）与 12-bit（0.02–0.12 ms + 长曝光）暗帧，但**没有任何暗帧落在
probe 的 hold-out 曝光档（0.051 / 0.028 ms）**，且预注册无冻结的暗场插值规则——按 §3.4
这些候选仍只能判 `not assessable`。**更正**：本文件初稿误称"归档完全没有暗场帧"，实际是
"暗场存在但与 probe 曝光/位深不匹配、且不可事后补"。
因此「补上游」= 补齐三件事：① 强度门评估器（✅ 本轮完成，离线验证）；
② **同会话 / 同曝光 / 同 bit-depth 的暗场**与六项前置（必须随新采集一并冻结，事后无法补）；
③ 一个真正过双门的冻结模型（**必须新采集带 train/holdout 拆分的数据**，现有归档无法制造合法
独立 holdout，且诊断显示模型前向误差本身偏大）。只有③通过后才回到 transfer `light/none` 采集。

---

## 1. 本轮已核事实（离线只读；未占用 SLM/CCD/激光；设备锁 `holder=None`）

| 事实 | 证据 |
|---|---|
| 无 checkpoint 过双门 | 方向门最好 9/14（`data/debug/offline_refit_20261118_freshep/direction_gate_2pass_results.json`，`axis_mismatch_validation_20261118/fit` 为 8/14，全部 `meets_10_of_14=False`）；强度门 held-out 全窗 relL2 0.51–0.72（`replicate_a` 0.7232 / `extend_b` 0.5055，`refit_provenance.json:metrics`），门线 δ=0.20 |
| 归档暗场不匹配（**非**"无暗场"） | 暗场库在 `data/ccd_configs/ccd/FJB24112232/darks/`：8-bit（0.024/0.043/0.047 ms）+ 12-bit（0.02/0.04/0.06/0.08/0.1/0.12/1/10 ms）。归档 probe 为 **Mono8 8-bit**，hold-out 曝光 `replicate_a`=0.051 / `extend_b`=0.028 ms —— **无暗帧落在该两档**，且无冻结插值规则 ⇒ 仍 `not assessable` |
| §2 强度指标此前缺实现 | 原代码仅有 `protocol.target_intensity_rmse`（整窗 unit-sum 归一化=**去尺度**）；P0 已新增 `strength_gate.masked_absolute_rel_l2`（掩膜内未缩放、可扣 DC），满足 §2 口径 |
| 可直接复用的实现件 | `fit.build_fit_model`（产出窗口 `phases/images/xy/rows`，`images` 已为 `count_rate`=counts/ms，`fit.py:134`）；`fit.fit_loss`/预测（train 拟合 `gain/background/bias_counts`，`prediction=optical*scale+offset+bias/exposure`，同 counts/ms，`fit.py:259-296`）；`fit.metrics` 的 `relative_l2=‖p−t‖/max(‖t‖,ε)`（`fit.py:152`，与 §2 同式）；`protocol.FixedAperture.box_mask/core_mask`（全幅像素坐标，`protocol.py:106-134`）；`artifacts.audit_run` 的窗口 mask 模式（`comps.xy`+`window_step` 在窗口网格上构造，`artifacts.py:65-75`）；`radiometry.count_rate` |
| 几何锚 | 每行 manifest 带 `spot_xy`（实测光斑）与 `exposure_ms`；模型窗口以 `reference_start` 光斑为中心、半跨=`span`、步长 2 |

---

## 2. 强度门评估器 a priori 冻结口径（先冻结，后计算）

> 纪律：以下口径在**跑任何候选之前**写定；不得在看到结果后调整，否则整份评估作废。

### 2.1 单位与输入

- **measured**：`comps.images[i]` = 该 probe 行的窗口帧经 `count_rate` 后的 **counts/ms**（原生 counts，不折位深）。
- **prediction**：冻结模型对该行 `phase` 的前向预测，走 `fit_loss` 同一条径，产出 **counts/ms**；
  radiometry（`gain/background/bias_counts`）**只在 dataset 的 `split=="train"` 行上拟合**，绝不使用 test/validation 行。
- 两侧单位一致（counts/ms），故单行 relL2 对曝光不变（分子分母同除 exposure）。

### 2.2 ROI / mask（v1，数据集无关，可复现）

- 窗口 = `build_fit_model(span=...)` 的 **`(span+1, span+1)`** 网格（`np.mgrid[-span:span+1:2]` 每 2 个全幅像素采样一次，故为 `span+1` 点/轴，`span=160` 即 161×161；**更正**：本节初稿误写 `2*span+1`，已按代码真实形状修正），步长 2，中心 = `reference_start` 光斑。
- 掩膜 = **整窗** 减去窗口中心处半径 `zero_radius`（窗口像素，由 `window_step` 折算）的**零级圆盘**。
  `zero_radius` 为本次冻结的会话级常量，必须写入证据 JSON；默认由该会话 `reference_start` 光斑
  几何确定并在冻结表中声明（不得用 test 结果反推）。
- 说明：v1 采用整窗（保守，覆盖全部成形光），理由是绝对强度保真门应看全视场；目标框（core）变体
  在几何可得的会话上作为**诊断副指标**报告，但不作门判据。

### 2.3 直流（暗场/背景）处理——§3.4 的硬约束

- **门判据要求真实暗场**：在 **probe 实际曝光档**上测得暗帧（关光/避激光，`capture_darks` 路径），
  从 measured 扣除；模型的电子偏置项 `bias_counts/exposure` 从 prediction 扣除，保留光学背景项。
- **缺暗场 ⇒ 该 pass 判 `not assessable`**（不是 fail，也不是 pass）。这是预注册 §3.4 的原文要求。
- 为可见性，另报一个 `median_fallback` 诊断数（measured/prediction 各自扣 `median`），
  明确标注 `diagnostic_only`，**永不参与门判据**。

### 2.4 指标与汇总

- 逐行（逐独立 holdout pass）：`relL2 = ‖p_sig − t_sig‖_mask / max(‖t_sig‖_mask, 1e-9)`，
  `p_sig/t_sig` 为扣直流后的 counts/ms。
- **每个独立 pass 单独计算**；采用**最差 pass** 判定（预注册 §4），不合并平均。
- 强度门：所有有效独立 pass 均 `relL2 ≤ δ=0.20`。
- 方向门：沿用既有判据（每模式 Pearson ≥ 0.50，独立 holdout 对中 ≥ 10 对通过），
  只对**锁定模型**的真实独立 holdout 计数；cross-fit 镜像对不重复计数（§4）。
- 联合放行：两门同时通过 **且** §3 六项前置有效；否则逐门报 `fail` 或 `not assessable`。

### 2.5 证据记录（每次运行必须写出）

- `checkpoint_sha256`、`dataset_manifest_sha256`、`script_sha256`（runner + 核心模块）、
  `span`、`window_step`、`zero_radius`、`bit_depth`、`full_scale`、`dark_source`（文件或 `missing`）、
  逐行 `{label, split, exposure_ms, relL2_gate|null, relL2_median_fallback}`、逐门 verdict、
  `validity`（六项前置逐项 ok/缺项）。
- 全部离线可复现，不打开设备。

### 2.6 P0/P1 实测确认的口径细节（实现时发现，均未改冻结阈值）

1. **原点选择影响数值**：只有模型原点 = 该 dataset 的 `reference_start` **实测点**（`build_fit_model` 默认，
   也是 `evaluate_transfer_on_new_session.py` 模板做法）时，归档 test 全窗 relL2 才复现（0.7232 / 0.5055）；
   用 checkpoint 里的 `origin`（fitted，如 `replicate_a` 的 `[1229.59, 1098.79]`）得 replicate_a `0.664707`。
   故 runner 默认 `--origin measured`，并在证据 JSON 记录该选择。
2. **checkpoint 存储的 radiometry ≠ 产生归档分数的 radiometry**：`replicate_a/refit.../frozen_parameters.json`
   存 `gain=47531.96 / background=0.0 / bias=0.679406`，而 `refit_provenance.json:metrics.refit.test.radiometry_fit`
   为 `37969.895 / 0.732569 / 0.730656`。按 §2.1 在 train 行重拟合 = 37969.895（与 provenance 一致），复现 0.7232；
   用 checkpoint 的 47531.96 则得 0.827。**含义**：日后若把该 checkpoint 当冻结模型，
   必须以「train 重拟合 radiometry」或「与 provenance 一致的导出」为准，不能直接吃 checkpoint 的 gain。
   runner 保留 `--radiometry {train,checkpoint}`，两者数值与来源都写进证据。
3. **归档 manifest 无 `bit_depth`/`full_scale`**：`replicate_a`/`extend_b` 证据里如实写 `null`，不臆造；
   validity 第 4 项按「暗场是否存在」判定。预注册 §3 第 5 项（settle/丢帧）离线不可判，标 `ok: null`。

---

## 3. 分阶段执行计划

| 阶段 | 内容 | 设备 | 产出 | 状态 |
|---|---|---|---|---|
| **P0** | 实现 `strength_gate` 核心指标 + 离线 runner + 单测 | 无 | 代码 + 测试通过 | ✅ 完成（15 单测 + ruff 全绿，见 §5） |
| **P1** | 对 archived 候选（`axis_mismatch_validation_20261118/fit`、`replicate_a`/`extend_b` refit）跑评估器 | 无 | 合规 verdict：预期 `not assessable`（暗场不匹配），median_fallback 诊断数 | ✅ 完成：`replicate_a`/`extend_b` 均 `not assessable`（全窗自检 0.723238/0.505481 精确复现）；`axis_mismatch` 未记录 `span`（grid 320）故未跑数值，仅记暗场不匹配→`not assessable` |
| **P2** | 暗场采集：按当次 probe 实测曝光档逐档配暗帧，纳入 session。**顺序已在 §6.0 改为 P3 之后**（曝光值采集后才可知） | 极小（关光/避激光） | `darks` 证据 | 待用户确认投入（规格已冻结，见 §6.4） |
| **P3** | 新测量：方向 holdout 两 pass（±Z5..Z11@0.05）+ 强度 train/holdout 数据集，**split 在捕获时写入** | 硬件（窗口） | 新 dataset | 待用户确认投入（规格已冻结，见 §6.2–6.3） |
| **P4** | 在 P3 的 train 上 fit 新 frozen 模型；在 holdout 上跑双门 | 无 | 资格 verdict | 待 P3 |
| **P5** | 资格通过 + 六项前置冻结 + 再次 dry-run 确认后，跑 transfer `light/none` 科学验收 | 硬件（窗口） | 验收数据 | 待 P4 |

> P2/P3 是「再采」的真正成本所在，且 P4 可能反馈模型类不足 → 需迭代 P3。这是跨会话研究轨道，
> 不是单窗口能完成。P0/P1 完全离线，先行交付。

---

## 4. 需用户拍板

1. **是否投入 P2/P3 的新测量轨道**（暗场 + 带独立 holdout 的新采集 + 可能多轮模型迭代）。
   这是拿到「双门合格冻结模型」的唯一合法路径；归档数据无法替代。
2. **暗场严格性（已按既定门槛锁定，非放宽项）**：本文件按预注册 §3.4 取「无同会话/同曝光/
   同 bit-depth 暗帧即 `not assessable`」的严格读法；`median_fallback` 仅作诊断证据，**永不**参与门判据。
   放宽需显式改预注册并记录理由——本项目既定要求是**不降低科学门槛**，故不建议。

---

## 5. 本轮变更与哈希

- 新增文档：`docs/dev/transfer_upstream_qualification_plan_20261119.md`（本文件）。
- 新增代码（P0，未提交 git；哈希为暗场量纲修复后的终版）：
  - `src/ao_shaping/tools/slm/transfer/strength_gate.py` — SHA256 `01178fa778ccaac4be3b243504802eb1f4c8b47bb4e9d291fa5345bdccabb46a`
    （`masked_absolute_rel_l2` + `window_zero_mask` + `STRENGTH_GATE_DELTA` + `gate_rel_l2`；逐行暗场匹配 + 位深/满量程校验终版）
  - `scripts/slm/evaluate_strength_gate_20261119.py` — SHA256 `1caca481443f6878e76ed47c44f7bba4f1af64ca73a97eac6f10da79d677dd8b`
  - `tests/ao_shaping/tools/slm/transfer/test_strength_gate.py` — SHA256 `743f7dd343920442f81c54a6a32de7bf51edd9dff065e8c25c4454107c3ebc04`，20 用例
  - `tests/ao_shaping/scripts/test_evaluate_strength_gate_20261119.py` — SHA256 `afb1795a4b801da893b3e2026f856c06d42de27fc72140f87e5e664a7d263315`，7 用例
  - 上述四文件合计 27 用例，2026-11-20 全通过（`pytest --import-mode=importlib`）。
- 依赖模块（参与证据哈希）：`fit.py` `bf01f26c…`、`protocol.py` `49cba3a6…`（见各证据 JSON `script_sha256`）。
- 新增代码（P3a 方向门，未提交 git；哈希为终版）：
  - `scripts/slm/direction_holdout_capture_20261119.py` — SHA256 `710e627bb162d451bc73a5f45d388e984b0d79dba06d6d04428d011b4e4a6fd8`
  - `scripts/slm/direction_holdout_gate_20261119.py` — SHA256 `5d2979313fcac3b7ccf6d8a0549ba212cf9a5a8cbde29f9789afd8b2aa4f945e`
  - `tests/ao_shaping/scripts/test_direction_holdout_capture_20261119.py`、`test_direction_holdout_gate_20261119.py` — 共 11 用例
  - 离线回归（2026-11-19）：方向门 `hybrid320_fresh_all` 复现 pooled 9/14、pass1 5/7、pass2 4/7、exit 1；`ruff` exit 0。

### 5.1 P2/P3 设备窗口与 P4 双门实测（2026-11-20）

设备窗口按 §6.1 冻结顺序串行执行，**全部正常关闭**（单进程、带 `--wait-until-idle 120`，未并行占用）：

| 步骤 | 输出目录 / 落点 | 结果 | 设备占用报告 |
|---|---|---|---|
| S0 六步光路探针（`--bit-depth 8`） | `data/debug/optical_path_probe/p3_upstream_20261120/20261120_012500Z/` | `probe_status=ok`、非饱和、离焦宽度 +19.2% / 峰值 +27.0%（SLM 真调制）、`bit_depth=8`/`full_scale=255` | SLM#1 + Daheng CCD + 激光；01:25:00Z→01:26:32Z；正常关闭 |
| P3a 方向 pass1 | `data/debug/slm_transfer/p3_direction_pass1_20261120/` | 24 行全 `ok`（14 `test` × Z5–Z11 ±0.05 + 10 calibration），exit 0 | 同上；09:27:46→约 09:31；正常关闭 |
| P3a 方向 pass2 | `data/debug/slm_transfer/p3_direction_pass2_20261120/` | 同上，独立重复测量，exit 0 | 同上；09:32:27→约 09:35；段间 60 s 冷却；正常关闭 |
| P3b 强度 train/holdout | `data/debug/slm_transfer/p3_zernike_20261120/` | 78 行（56 train + 12 test + 10 calibration），全 `0.087 ms`、Mono8/8-bit、全 `ok`，exit 0 | 同上；09:37→09:46:37；正常关闭 |
| P2 逐档暗场 | `data/ccd_configs/ccd/FJB24112232/darks/dark-exp-0.087ms.npz` | `0.087 ms`/8-bit/255、16 帧、`mean=0.3215`/`peak=2.0`（纯读出噪声）；用户确认挡光后采集 | **仅 Daheng CCD**（SLM/激光未开，已物理挡光）；09:52:08→09:52:13；正常关闭，`DeviceLock` 已释放 |

P4 双门评估（离线，设备已释放；用 P3 train 拟合、holdout 判门，未改模型/阈值）：

- P4(a) `refit_transfer_forward.py`：`p3_fit_20261120/frozen_parameters.json`，`checkpoint_sha256=380b9b87885fe70a4279dbe00fc500dcc3893b9422ddc84ccd0115e491c6619a`（train-only fit 门禁全绿，选中迭代 209）。裸 loss 诊断显示 refit 在 test 上劣于基座（relL2 0.7866 vs 0.7188，`refit_beats_base_on_score_split=false`）。
- P4(b) 强度门 `evaluate_strength_gate_20261119.py` → `p3_gate_20261120/strength_gate.json`：**`verdict=fail`**（exit 1）。12/12 holdout 行全部可评估（暗场逐行匹配 `0.087 ms`、位深 8/满量程 255、六项 validity 前四项全 `ok`），最差 `relL2=0.8849` ≫ δ=0.20。属**有效 fail 判决**，非 `not_assessable`。
- P4(c) 方向门 `direction_holdout_gate_20261119.py` → `p3_gate_20261120/direction_gate.json`：**`verdict=fail`**（exit 1），pooled 4/14（需 10），pass1 2/7、pass2 2/7。`checkpoint_sha256` 与强度门一致。
- **结论：双门均未通过，transfer `light/none` 科学验收继续暂停，不进入 P5。** 按 §6.8，只有双门同时通过且用户再次确认才恢复；本次为真实科学结论，未以调阈值/换模型方式操纵通过。

---

## 6. P2/P3 a priori 冻结方案（先冻结、后采集；本节写定于设备窗口之前，未占用设备）

### 6.0 侦察修正：三项必须先落进方案的事实

| 事实 | 证据 | 对方案的影响 |
|---|---|---|
| 曝光是**逐行自适应**，不是会话常量 | `replicate_a` manifest：`train` 行同时存在 0.027 与 0.051 ms；`extend_b` 为 0.028 与 0.051。`capture.capture` 每行以 `start_ms=上一行曝光` 跑 ladder，`exposure_max_steps=2` | 暗场必须**逐档**匹配；且暗场只能在本会话 probe 跑完后、按 manifest 实测值采集 → **P2 顺序后移到 P3 之后** |
| 生产采集口是 Mono8 **8-bit**，且 `capture.py` CLI **没有** `--bit-depth` | `_resolve_capture_exposure` 只读 `getattr(args,"bit_depth",None)`；`capture.py` 的 13 个 `add_argument` 中无 `--bit-depth`；`zernike_probe_hardware.py` 的 namespace 也不含该键。归档 `_optical_probe.json: bit_depth=8` | P3 数据 = 8-bit / 满量程 255。六步光路探针**默认** `--bit-depth 12`（`S0_BIT_DEPTH`），必须显式改 `--bit-depth 8`，否则 session record 记的是与数据不同的模式 |
| 方向门 holdout **无现成采集脚本** | `zernike_probe_hardware.py` 的 train 是 ±Z4..Z11@±0.1/±0.2、holdout 是 `probe_grid` 随机相位，**不含 ±Z5..Z11@0.05**；历史两 pass（`current_epoch_zernike_direction_20261118{,_pass2}`，`config.json` 记 `script_sha256=a5f4b85d…`）在 `scripts/ .worktrees/ src/` 共 4486 个 `.py` 中**无匹配文件** | P3a 需新增专用采集脚本，写定后**先冻结哈希**再 `--execute` |

### 6.1 设备窗口顺序（硬约束：单窗口串行，禁止并行占用）

```
S0 六步光路探针 (SLM+CCD+激光)
  → P3a pass1 (SLM+CCD+激光)
  → P3a pass2 (SLM+CCD+激光)
  → P3b 强度数据集 (SLM+CCD+激光)
  → P2 逐档暗场 (仅 CCD；关光/避激光)
  → 释放
```

理由：曝光档只有 P3 跑完才可知（6.0 第 1 行），暗场必须后置。每步都要报**占用设备 / 操作 / 释放时刻 / 是否正常关闭**；
任一步失败：该步输出目录作废、**不复用**，重跑从该步起；跨步不复用（避免光路漂移混入）。

### 6.2 S0 六步光路探针（产出 session record）

```
.venv\Scripts\python.exe scripts\slm\probe_optical_path_validity.py ^
  --session p3_upstream_<YYYYMMDD> --bit-depth 8 --wait-until-idle 120 --no-plot
```

- **冻结 `--bit-depth 8`**（偏离 SKILL 默认 12 的唯一理由：生产 transfer capture 走 Mono8，探针必须记录与数据同模式）。
  代价：8-bit 下 uniformity/线性度结论被量化噪声主导 —— 本步只取「设备在线 / 光路有效 / 曝光不超量程 / SLM 真在调制」四项，
  并在记录里写明这一让步；不据 8-bit 记录宣称均匀度结论。
- 产出：`<out-root>/<session>/<UTC>/optical_path_record.json`（`devices.camera.bit_depth=8`、`full_scale=255`、
  `exposure_range_ms`、`exposure_check`、`occupancy`）+ `optical_path_probe.json`。路径与 `sha256` 写入 P4 证据。
- **前置判定**：`probe_status` 非 ok、`occupancy.closed_normally != true`、`exposure_check.saturated == true`、
  或 SLM 未真正调制 → **停止**，不进 P3。

### 6.3 P3a 方向 holdout（±Z5..Z11@0.05，两 pass）

**为什么不能用 `zernike_probe_hardware.py`**：它的 train 覆盖 ±0.1/±0.2、holdout 是随机相位；而历史方向门判据
（每对 Pearson ≥ 0.50、10/14）是在 **amp=0.05** 上标定的。换振幅等于换判据工作点，属于事后改门，禁止。

新脚本**已实现**：`scripts/slm/direction_holdout_capture_20261119.py`，SHA256
`710e627bb162d451bc73a5f45d388e984b0d79dba06d6d04428d011b4e4a6fd8`（与 `zernike_probe_hardware.py` 同构：
patch `capture.probes` → 调 `capture.capture`）：

- 探针序列：原 calibration 段（`reference_start` + `ramp_x/y` 64/128 ±1，共 9 行）→ **Z5..Z11 各 ±0.05（14 行，
  `split="test"`，label `holdout_{i:03d}_Z{mode}_{sign}{amp}`）** → `reference_end`。旧的 `_Z{mode}_{sign}{amp}`
  后缀必须保留，因为评估器用 `_Z(\d+)_([+-])(\d+\.\d+)$` 配对。`build_plan(modes, amp=0.05)` 为纯函数，便于离线核对。
- 冻结参数：`--grid 128 --frames 2 --seed 1104229 --exposure-max-steps 2`（与历史两 pass 一致；曝光种子用 capture 的 `auto`）。
- 默认 plan-only（不接触硬件），`--execute` 才取锁实验；`--wait-until-idle 120`，沿用 `device_mutex` 既有机制。
- 两次 pass 参数完全相同（同一批相位、独立测量）；`--out` 为全新目录，`exist_ok=False`：
  `data/debug/slm_transfer/p3_direction_pass1_<YYYYMMDD>/`、`..._pass2_<YYYYMMDD>/`。
- **已验证（2026-11-19，离线）**：plan-only 打印 14 行 `holdout_*` 全 `split=test`，exit 0；无 `--execute` 不取锁。
- **口径透明化**：两 pass 是**同一批 7 个相位对的独立重复测量**，14 对中每个模式计 2 次。这是历史判据（10/14）
  的原始口径，本方案**不重定义**判据；证据 JSON 必须同时报 `per_pass`（各 7 对）与 `pooled`（14 对）两个计数，
  避免 14/14 被误读成 14 个不同相位。
- 写入即声明：这两份数据**永不进入 fit**（P4 只喂 `p3_zernike_*` 的 train 行）；P4 证据里显式列出这两个目录及其 `manifest_sha256`。

### 6.4 P3b 强度数据集（train / holdout 独立拆分）

先 dry-run（无 `--execute` 只打印计划），确认目录全新、参数与下表一致：

```
.venv\Scripts\python.exe scripts\slm\zernike_probe_hardware.py ^
  --out data\debug\slm_transfer\p3_zernike_<YYYYMMDD> ^
  --modes 4 5 6 7 8 9 10 11 --amplitudes 0.1 0.2 ^
  --n-random 24 --n-holdout 12 --frames 3 --seed 1104229 --wait-until-idle 120
```

- 冻结：`split` 由 generator **在捕获时**写死（`ztrain_*`/`rtrain_*` = train，`holdout_*` = test）；预期 10 calibration +
  56 train + 12 test 行（与归档 `replicate_a` 同构，便于逐项对照）。
- 独立性：holdout 相位源自 `probe_grid(1000+i, seed+5)`，train 随机变体源自 `probe_grid(i, seed+777)`，不同族；
  holdout 行不参与 fit、也不参与 radiometry 拟合（`--radiometry train` 只吃 `split=="train"`）。
- 该目录也是强度门的 `--dataset`，其 `_optical_probe.json`（`bit_depth=8`）是数据集位深的**唯一**来源。

### 6.5 P2 暗场（P3 之后，仅 CCD）

1. **档位来源**：P3b（以及 P3a，若其行也要进任何强度比较）manifest 中**所有 split** 的 `exposure_ms` 去重
   （含 calibration 行；不能只看 test）。每个值按 manifest 原样 repr 传回，不四舍五入。
2. 关光 / 避激光后：
   `.venv\Scripts\python.exe -m ao_shaping.tools.ccd.capture_darks --exposures-ms "<逗号分隔原值>" --frames 16 --bit-depth 8 --cam-type daheng --cam-id 0`
3. 落点固定为 `data/ccd_configs/ccd/<serial>/darks/`（`capture_darks.py` **没有** `--out`，只有 `--config-root`；不得另建目录）。
4. **匹配规则（冻结）**：用暗场 manifest 的**设备回读生效曝光**与行 `exposure_ms` 比较，判据 `abs(diff) <= 1e-6`；
   同行要求 `bit_depth=8` 与 `full_scale=255` 一致。**不引入 `rel_tol` 放宽**；相邻档代替、插值、事后重标一律禁止。
5. 无匹配档 ⇒ 该 holdout 行 `not assessable`（预注册 §3.4）。**缺一档就整份门判 `not assessable`，不得只报有暗场的那几行。**

### 6.6 行有效性（settle / 丢帧 / 饱和）

- 沿用 `capture.py` 内建行为：曝光阶梯每档 0.6 s settle（`capture.py` 阶梯注释），每行取 `--frames` 帧均值；
  `discard_frames_after_exposure_change` 由相机/驱动决定并记在 optical path record，不在本方案另加丢帧。
- 行有效 ⇔ `exposure_setting.status == "ok"` ∧ `saturation_pixels == 0` ∧ `peak < 250/255 × full_scale`。
- 任一 holdout 行无效 ⇒ 该 pass `not assessable`（不是 fail）。

### 6.7 ROI / mask / 窗口（P4 判据用的冻结常量）

- 窗口：`--span 160`（`(span+1)²=161²`，步长 2 → 覆盖 320 sensor px 见方），与 P1 复现口径一致。
- mask：整窗减去零级圆盘 `--zero-radius-px 4`（窗口像素 = 8 sensor px）。
  **a priori 理由**：零级为衍射极限斑，其 3σ 半径远小于 8 sensor px；固定值不含任何 test 信息。
  卫星：若该会话 `reference_start` 的二阶矩半径 > 8 sensor px，则该 pass `not assessable`（几何前置不满足），
  **不得事后调大 `zero_radius`**。
- 原点与辐射度：`--origin measured`（dataset 的 `reference_start` 实测点）、`--radiometry train`（只在 train 行拟合），
  理由与 P1 复现一致性见 §2.6。

### 6.8 P4 双门评估（命令冻结；仅在 P3 数据齐备后执行）

```powershell
# (a) 在 P3 train 上拟合冻结模型：协议逐字沿用归档 refit，只换 dataset
.venv\Scripts\python.exe scripts\slm\refit_transfer_forward.py ^
  --dataset data\debug\slm_transfer\p3_zernike_<YYYYMMDD> ^
  --checkpoint data\debug\slm_transfer\20261118_transfer_mil_zernike_probe_v1\fit_m3_full\frozen_parameters.json ^
  --out data\debug\slm_transfer\p3_fit_<YYYYMMDD> ^
  --span 160 --adaptation full --radiometry exposure --iterations 220 --patience 999 --min-delta 0

# (b) 强度门（holdout = p3_zernike 的 test 行；暗场逐档；位深 8）
.venv\Scripts\python.exe scripts\slm\evaluate_strength_gate_20261119.py ^
  --dataset data\debug\slm_transfer\p3_zernike_<YYYYMMDD> ^
  --checkpoint data\debug\slm_transfer\p3_fit_<YYYYMMDD>\frozen_parameters.json ^
  --dark-dir data\ccd_configs\ccd\<serial>\darks ^
  --optical-path-record <S0 记录路径> --data-bit-depth 8 --data-full-scale 255 ^
  --span 160 --window-step 2 --zero-radius-px 4 --origin measured --radiometry train ^
  --out data\debug\slm_transfer\p3_gate_<YYYYMMDD>\strength_gate.json

# (c) 方向门（两 pass × 7 模式 = 14 对；每对 Pearson ≥ 0.50，≥ 10/14 通过）
#     评估器 scripts/slm/direction_holdout_gate_20261119.py
#     （SHA256 5d2979313fcac3b7ccf6d8a0549ba212cf9a5a8cbde29f9789afd8b2aa4f945e；数学逐字移植归档 2-pass 脚本，
#       CROP=128、MODES=5..11、每 pass 全行均值椭圆中心、direction_agreement、rate=mean/exposure_ms 全部保留）
#     --runs 指向 p3_direction_pass1/pass2，--checkpoint 指向 (a) 的 frozen_parameters.json
#     已验证：用归档两 pass + hybrid320_fresh_all 复现 pooled 9/14、pass1 5/7、pass2 4/7，exit 1（fail），与归档证据一致。
```

- (a) 的 `--checkpoint` 是**初始化基座**（归档 refit 用的同一个 M3 全拟合，`sha256=4c642005…`），不是评估对象；
  所有测量决策只落到 P3 train 行。
- (b) 的判据：所有有效 holdout 行成立且 `max relL2 <= δ=0.20`；退出码 0 pass / 1 fail / 2 not assessable。
- 证据必须含：模型 `checkpoint_sha256`、`dataset_manifest_sha256`、`script_sha256`（评估器 + 采集脚本）、
  S0 记录 `sha256`、暗场清单、六项 validity。
- **只有 (b) 与 (c) 同时通过，且用户再次确认**，才回到 transfer `light/none`（P5）。

---

## 7. 仍未定的两件事（不阻塞本轮文档冻结，阻塞 P4 执行）

1. **P3a 采集脚本与方向门评估器已完成**（§6.3、§6.8(c)）：两者均已写入仓库并冻结哈希，2026-11-19 离线回归通过
   （方向门复现归档 9/14 / exit 1；11 单测通过；`ruff` exit 0）。剩余阻塞点是**设备窗口未开**，不是工具缺失。
2. **`capture_darks.py` 无 `--wait-until-idle`**：它用 `DeviceLock.acquire(timeout=60.0)`（`LOCK_TIMEOUT_S`）自带的固定等待预算，
   没有可调参数。执行时先确认 `holder=None`，否则等 60 s 满即失败退出；不得与任何 probe/capture 并行。

- 验证：`pytest …test_strength_gate.py -q --import-mode=importlib` → 15 passed；`ruff check` 三文件 → 全绿；
  自检 `replicate_a`(span160) 全窗 mean `0.723238` vs 目标 `0.7232375`（差 ~5.7e-9）、`extend_b` `0.505481` vs `0.5054810`（差 ~1.6e-8）。
- 暗场量纲修复（本轮）：`dark_rate` 已是 counts/ms，旧代码 `measured - dark_rate * exposure` 会少减 `1/exposure` 倍
  （0.051 ms 处约 20 倍）。抽出 `gate_rel_l2(pred, meas, mask, bias_rate=, dark_rate=)` 统一两处口径，
  新增 2 个量纲 pin 测试（等于 `meas - dark_rate`、不等于 `meas - dark_rate*exposure`；支持逐像素 map）。
  另加两条硬门槛：暗 npz/json 只有原始 counts 而无 `exposure_ms` → `DarkUnavailable`（不再静默当 counts/ms）；
  暗帧记录的曝光 ≠ probe 曝光 → `not_assessable`。冒烟验证：`dark-exp-0.024ms.npz` → 明确 `carries no exposure_ms`；
  合成 0.024 ms 暗 + 0.051 ms probe → `dark exposure 0.024 ms != probe exposure 0.051 ms`；
  合成同曝光零暗（仅验证接线，非证据）→ 12/12 行算出 gate、worst `relL2=1.0362`、verdict `fail`，证明判定路径可执行。
- P1 证据（新增，`data/` 已 gitignore）：`data/debug/slm_transfer/strength_gate_eval_20261119/replicate_a.json`、`…/extend_b.json`，两者 verdict `not_assessable`（暗场与曝光不匹配，exit 2）。
- 设备纪律：本轮纯离线只读 + 离线运行评估器，未占用 SLM/CCD/激光；最近真实设备操作为 #238 run4（释放证据 12:18:40Z）。

---

## 8. 执行记录（2026-11-20：P2/P3 采集 + P4 双门有效 fail）

按 §6.1 冻结顺序单进程串行执行，每段带 `--wait-until-idle 120`、全新 `--out`，`try/finally` 正常释放，
无并行占用。详细逐段表见 `agent_notes_20261120.md` §9。

| 段 | 产物 | 结果 |
|---|---|---|
| S0 光路探测 | `data/debug/optical_path_probe/p3_upstream_20261120/20261120_012500Z/` | `probe_status=ok`、非饱和、离焦宽度 +19.2% / 峰值 +27.0%（SLM 真调制）、8-bit/255 |
| P3a 方向 pass1/pass2 | `data/debug/slm_transfer/p3_direction_pass{1,2}_20261120/` | 各 24 行全 ok（14 test × Z5–Z11 ±0.05 + 10 calibration），段间 60 s 冷却 |
| P3b Zernike 拆分采集 | `data/debug/slm_transfer/p3_zernike_20261120/` | 78 行（56 train / 12 test / 10 calibration），全 0.087 ms、Mono8/8-bit、全 ok |
| P2 暗场 | `data/ccd_configs/ccd/FJB24112232/darks/dark-exp-0.087ms.npz` | 16 帧 peak=2.0 纯读出噪声，与 P3 曝光逐档匹配；仅开 CCD，SLM/激光未开，09:52:13 正常释放 |
| P4(a) refit | `data/debug/slm_transfer/p3_fit_20261120/frozen_parameters.json` | `checkpoint_sha256=380b9b87…`，train-only fit 门禁全绿，选迭代 209；裸 loss test 0.7866 劣于基座 0.7188（如实记录，未换模型） |
| P4(b) 强度门 | `data/debug/slm_transfer/p3_gate_20261120/strength_gate.json` | 12/12 可评估，最差 `relL2=0.8849` ≫ δ=0.20，`verdict=fail`（exit 1） |
| P4(c) 方向门 | `data/debug/slm_transfer/p3_gate_20261120/direction_gate.json` | pooled 4/14 < 10（pass1 2/7、pass2 2/7），`verdict=fail`（exit 1） |

**判定（冻结口径、不得事后择优）**：双门均为**有效 fail**（非 `not_assessable`），
未调阈值、未换 checkpoint。按 §6 与 `transfer_strength_gate_preregistration_20261119.md`，
transfer `light/none`（P5）继续暂停，不进 P5、不启动其硬件 `--execute`。
离线抢救路线调查（`agent_notes_20261120.md` §10）显示：纯 train loss 选点 iter209 与 test 最优 iter18
（relL2 0.5500）均 ≫ δ=0.20；换迭代最优仍过不了门，合法早停须重采（无 validation split）；
P3b 78/78 无饱和、曝光窗口窄至约 0.10 ms 上限；方向门 4/14 更像 checkpoint 问题
（新采 model_phase 与归档 14/14 完全相同）。**无任何离线项可把 fail 翻成 pass**，故维持暂停。

动态整形速度解耦实验（同日、独立工作线）中的 model 相位仅作诊断（`model_phase_diagnostic_only=true`），
不得当作 P4 资格通过证据（见 `dynamic_decoupling_experiment_plan_20261120.md` 重要边界）。
