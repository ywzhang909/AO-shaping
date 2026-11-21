# 下一步实验计划与采集数据方案（2026-11-20 冻结稿）

> 本文是 `hw_window_20261120b_issue247_238_results.md`（本轮结果）之后的**下一步**编排，覆盖
> 离线分析线与需要设备窗口的采集线。**本文不占用设备**；任何 `--execute` 段须用户另行明确批准，
> 并在新窗口内按 §9 报告设备占用与释放。设备窗口内**一次 open/close**，段间冷却 ≥200 s。

## 摘要

本轮硬件窗口（2026-11-20b）得到两个可复用的确定结论与两个未闭合项：

- **#247 衍射探针本身可靠**：级间距严格 ∝ 1/周期，单向 blaze 梯单侧成立，tilt 正负反向成立。A0 离线核验已证实 grating/tilt/ladder 的相位梯度均符合 `x=列、y=行`，**不是 `axis=1/0` 标签错位**。但原始帧中 grating/tilt 两轴都落 CCD-x，`ladder_x` 落 CCD-y，数据之间仍不自洽；不能冻结物理映射。`camera_pixel_size_um=2.2` 与 `slm_pixel_pitch_um=8.0` 在仓库已有权威配置，缺的是将它们和实际光学标度绑定的可信 `focal_magnification`/系统标定记录。
- **#238 低漂移复采成功但不够正式**：ABBA 段内前后参考变化 <0.1%（run4 的 ±31% 未复现），地标峰 ≈448 高度重现；但仅 **4 帧/灰度且无遮光暗场**，不满足 §6.6 永久报告门槛，不发布新 g₂π。
- **#234/#236 双门 fail**：强度门 12/12 最差 `relL2=0.8849` ≫ `δ=0.20`；方向门 pooled `4/14`（需 10）。transfer `light/none` 继续暂停，**不得**用同口径硬件验证 #234/#236。

因此下一步的优先级不是"再采一次"，而是：**先离线把两个采集脚本的缺口补齐（零成本、可能直接解释矛盾），再用一个合并窗口做 #247 配准与 #238 正式 g₂π；#234/#236 先离线修门，门过再谈硬件验证。**

> **同日已完成、与本计划并行的工作线**（不属下面 A0/B0/C/D 队列）：
> - transfer 上游资格 P2/P3 采集 + P4 双门已执行完，双门均有效 fail（12/12 最差 `relL2=0.8849`；方向 4/14），
>   证据与离线抢救调查见 `transfer_upstream_qualification_plan_20261119.md` §8、`agent_notes_20261120.md` §9–§10；
>   transfer `light/none` 暂停判定与本计划 §0 一致。
> - 动态整形速度解耦采集 T1–T5 已完成（30/30 ok，实测约 2.3 fps，瓶颈 SLM 写入+换槽中位 0.312 s），
>   正式报告 `report/shape_trials/speed_decouple_capture_20261120.md`，冻结方案
>   `dynamic_decoupling_experiment_plan_20261120.md`；其 model 相位仅诊断，不作为 P4 翻案证据。

## 0. 现状与门控状态

| Issue | 当前结论 | 阻塞 / 缺口 | 下一步类型 |
|---|---|---|---|
| #247 | 探针可靠，映射未冻结 | 轴映射矛盾；缺 pitch/放大率/配准图样 | **离线 A0 → 实验 A1** |
| #238 | 低漂移可复现，非正式 | 代码支持已齐（B0）；**照明关断 hook 未接线**；4 帧/灰度；无第三周期点 | **离线接线 hook → 实验 B1** |
| #234/#236 | 方向/问题有效，收益未证 | 双门 fail（强度 0.8849、方向 4/14） | **离线 C，不实验** |
| #235 | 零级定心已验证；LUT/shift_phase/pupil（瞳面）-center 未闭环 | 物理 LUT 与独立 pupil（瞳面）-center 未验收 | 离线 D，实验可选 |
| #249/#250 | 有效，核心修复已落地 | 待 owner 复核 | 离线守护 |
| #251 | 14 红清零，4 个系统性缺陷未根治 | `.worktrees/issue251` 有并行 WIP | 离线，勿与主树混提交 |

## 1. 总原则

1. **离线优先**：任何设备动作之前，先做能离线做的检查（相位轴约定、常量清点、脚本 dry-run）。零成本检查优先于实验。
2. **一次窗口合并**：同一窗口内按 §8 顺序执行 A1→B1，SLM/CCD 各 open/close 一次，段间冷却 ≥200 s（背对背 open/close 会触发驱动板假死）。
3. **冻结口径、不得事后择优**：灰度层集合、周期点、帧数、曝光、暗场、ABBA 顺序在采集前写死；分析只按冻结规则。
4. **溯源与释放**：唯一新 `--out`（`mkdir(exist_ok=False)`）、`DeviceLock` + `--wait-until-idle`、六步光路探测、写 `optical_path_record.json`（`ao-optical-path-record/v1`）与 `released.json`；`finally` 中 `set_grayscale(0)→cam.close→slm.close→lock.release`。
5. **不越界发布**：无新配对帧溯源，不宣称 #234/#236 收益；不冻结 #247 物理映射；不发布 #238 新 g₂π；不写 `slm_geometry_calib.npz`。

## 2. 优先级、依赖与窗口编排

```
A0 (离线) ──► A1 (#247 配准, ~5 min) ──┐
B0 (离线) ──► B1 (#238 正式 g2π, ~4 h) ─┴─► 同一设备窗口, 段间冷却
C  (离线双门修复) ──► 门过且有裕度 ──► 才可申请 #234/#236 上机
D  (离线 LUT/pupil-center) ──► 可选小采集
```

- **A0 与 B0 可并行**（各自独立脚本，互不写同一文件）。
- **A1 必须先于 C**：方向门的方向性正受 #247 面板↔CCD 映射影响；映射未冻结前，方向门的 fail 无法归因。
- **B1 依赖 B0 的脚本改动**；B0 未完成则窗口只做 A1。

## 3. 实验 A：#247 面板↔CCD 轴映射冻结

### 3.1 目的与判据

用**无 180° 歧义**的配准图样，解出 2×2 仿射映射 `M`（panel 位移 px → CCD 位移 px），并判定其是否可冻结。

| 指标 | 定义 | 接受条件 |
|---|---|---|
| 残差 RMS | 所有位移点 `‖M·d_panel − d_ccd‖` | ≤ 1.0 px 且相对 ≤ 2% |
| 条件数 | `cond(M)=σ_max/σ_min` | 报告；各向异性比 `σ_max/σ_min` 须与实测一致 |
| 旋转角 | `atan2` 主方向 | 报告；与历史 `+88.544°`（180° 对称族）对照，不强行套用 |
| 符号自洽 | tilt ±、grating ± 的反向 | 全部满足 |

**判据**：残差小且 10 个探针方向自洽 → 写 `issue247_axis_map.json`（建议值，**不写 npz**）；否则记"映射仍歧义"，不冻结。

### 3.2 前置 A0（离线，零设备）— 已完成，2026-11-20

**A0-1 相位轴数值核验（已完成）**：直接调用生产 `grating_phase` / `tilt_phase` / `ladder_phase`，对 `(rows,cols)` 计算 `|d/dcol|` 与 `|d/drow|` 梯度总量；`grating_x`、`tilt_x`、`ladder_x` 均以列梯度为主，三个 `y` 变体均以行梯度为主。`tilt(+)==-tilt(-)` 的最大误差为 0。又在 `(1200,1920)` 面板上作离线 2D FFT：`grating_x` 只沿 x 方向出级，`grating_y` 只沿 y 方向；`ladder_x/y` 也分别沿对应面板轴出级。**结论：生产相位构造与标签一致；“axis 标签写反”假设否定，不改标签。** 新增回归测试 `test_named_pattern_axes_match_panel_column_row_semantics`，定点测试 17 passed。

**A0-1b 数据复核（已完成）**：历史 `plan.json` 的 11 个相位哈希与当前 `build_patterns()` 全部一致（ID 集、SHA256 均一致），排除 plan 与当前代码版本不一致。原始帧独立重分析结果：`grating_x` ±95/96 CCD-x、`grating_y` ±60 CCD-x、`tilt_x` ±95 CCD-x、`tilt_y` ±62/65 CCD-x；而 `ladder_x` 的 +1/+3 在 CCD-y（−88/−232），`ladder_y` 在 CCD-x（+78/+235）。因此冲突是**采集响应/相位上屏链路与单轴坐标映射之间**，不是分析中把 x/y 标签读反；仍需进一步定位，不能冻结映射。

**A0-2 常量清点（已完成）**：`camera_pixel_size_um=2.2` 来自 `data/ccd_configs/ccd/models.json`（MER2-507-23GM）；`slm_pixel_pitch_um=8.0` 来自 `src/ao_shaping/utils/slm/bench_constants.py::SLM_PITCH_M`。按 λ=1064 nm、两者代入光栅公式，当前观测分别反推有效焦距：grating_x p128/p256 ≈202/201 mm；grating_y p128/p256 ≈127/125 mm。两轴结果不相容，**这些只是反推诊断值，不是实测焦距，不可写进标定常量**。历史 `TILT_SHIFT_SCALE=7400` 及报告中的约 122.4 mm 不足以解释 x 轴约 202 mm 的结果；当前首要问题不是缺 2.2/8.0 两个尺寸常量，而是两轴尺度来源冲突。

**A0-3 下一步离线准备**：暂不加“普通十字/单点”图样——它们不能单独区分“相位上屏/光学轴交换”与“读出坐标映射”。先加离线合成验收：对候选的单轴平面相位、二维非对称标记/二维光栅及 ladder patch 布局作 FFT，输出可区分的有符号 spot 预测与哈希；只在该预测验证通过后，才准备相应采集图样。不得仅凭现有数据挑一个仿射矩阵。历史 `plan.json` 哈希已复核全 11 项一致。

### 3.3 实验采集 A1（冻结参数）

| 项 | 值 |
|---|---|
| 脚本 | `scripts/slm/issue247_geometry_capture_20261120.py`（待 A0-3 离线合成验收通过后，才确定是否扩展） |
| `--out` | `data/debug/hw_window_<date>c/issue247_axis/`（唯一新目录） |
| `--exposure-ms` | `0.05`（沿用本轮探测：flat 峰值 1717 counts ≈42%，未饱和） |
| `--settle-s` | `0.8` |
| `--saturation-fraction` | `0.9` |
| 帧/图样 | 4（均值降噪，沿用本轮口径） |
| 图样 | 既有探针 + A0-3 离线验收后选定的非对称配准图样（清单采集前冻结） |
| 设备 | Santec SLM-200 #1（1064 nm，`two_pi_grayscale=993`）+ Daheng MER2-507-23GM（12 bit 全幅，不物理开窗） |

```powershell
.venv\Scripts\python.exe scripts/slm/issue247_geometry_capture_20261120.py `
  --execute `
  --out data/debug/hw_window_<date>c/issue247_axis `
  --exposure-ms 0.05 --settle-s 0.8 --saturation-fraction 0.9 `
  --wait-until-idle 900
```

### 3.4 解算与接受

- 对每个图样取 4 帧均值、平滑、2D 局部极大 + 行列投影双验证，定位各级/单点位移（延续 `i247_spots2d.py` 口径）。
- 用 ≥6 个位移点最小二乘解 `M`；报告 `σ_max/σ_min`、旋转角、残差 RMS、逐点残差图。
- 判据见 §3.1；满足则输出 `issue247_axis_map.json` 并更新 #247 评论（**不改 issue 状态、不 push**，除非授权）。

### 3.5 不做什么

- 不写 `slm_geometry_calib.npz`。
- 不把单次映射当成"物理常数已定"；`focal_magnification` 未实测时标注"反推"。
- 不在映射未冻结前重判方向门。

## 4. 实验 B：#238 正式 g₂π 标定

### 4.1 目的与判据

在**低漂移 + 遮光暗场 + 每灰度 20 帧 + 第三周期点**的口径下重估 g₂π，使系统误差可闭合。

| 指标 | 目标 |
|---|---|
| 每灰度帧数 | 20（run4 口径） |
| 周期点 | 32 / 48 / **64**（3 点，至少能检验周期趋势而非两点外推） |
| 顺序 | palindrome（`32 48 64 64 48 32`） |
| 功率参考 | ABBA，每 block 前后各 ≥4 帧 flat |
| 遮光暗场 | 每 block 前后各 20 帧，同曝光 0.08 ms |
| 数据曝光 | 固定 0.08 ms；bracket 0.02/0.04/0.08/0.16 仅审计饱和 |
| 2π 口径 | `g_ref=1023`（编码）、`device_two_pi_gray_readback=993`（物理）、`two_pi_gray_true` 由离线拟合给出 |

### 4.2 前置 B0（离线：暗场采集扩展）— 代码已完成，2026-11-20

**结论**：脚本侧暗帧采集已实现并测试通过；但**采集端仍缺照明关断控制**（本 bench 路径无激光快门 / SLM blank / 相机快门的 API），故 B1 的 `--dark-frames 20` 目前会 fail-fast，须先补硬件 hook 才能真采。B0 把「采集缺口」从数据里彻底诚实化，而不是假装采了暗场。

**已实现（`scripts/slm/issue238_gray_phase_probes.py`）**：

1. 新增 `--dark-frames N`（默认 0）与 `--dark-at {pre,post,both}`（默认在 N>0 时为 `both`）。暗帧插在每个 sweep block 的 `pre` 参考之前、`post` 参考之后，带 `block_index`/`block_id`。
2. 暗帧标记 `role="dark"`、`illumination="blocked"`、`slm_pattern_written=False`、`panel_gray_sha256=None`（不写 SLM 图样，故不伪造面板摘要）；帧格式仍 uint16 + Mono12 满量程 4095 + 同 `save_frame` 校验。
3. **暗帧只记录、不扣除**：`summary.json.dark.dark_subtracted=False`，逐帧总量、中位、总体 σ、帧文件清单；`summarize_power_reference` 与 `saturated_frames` 明确排除 `role="dark"`。**这是与旧稿的关键差异**：原稿写 `dark_subtracted=true`，实际实现为记录（见下方缺口），不得据此外宣称已完成暗场扣除。
4. **单曝光规则**：暗帧必须与所选 sweep 曝光完全一致；出现第二曝光立即 raise → 写 `dark_abort.json`（同时保留 `power_reference_abort.json`，若该轨开启），并正常释放锁。
5. **fail-fast seam**：`--dark-frames N>0` 但未提供 `_run(dark_capture_hook=...)` 时，在 import/打开驱动**之前**抛错并写 `dark_abort.json` —— 明确「无照明关断无法采暗场」，不虚构硬件调用。
6. **默认零回归**：`--dark-frames 0` 时 plan 无 `dark` 段、帧序列与旧调用逐帧一致。

**B1 窗口 wrapper 透传（`scripts/experiments/run_issue238_period_sweep_20261119.py`）**：B1 计划点名的 runner 是该 wrapper，它原先不透传暗场标志（§4.3 的命令会失败）。已补 `--dark-frames` / `--dark-at` 并只在显式启用时转发（缺省 argv 与历史窗口逐字一致）；`build_segments` 每段 `--mode binary` 下携带该标志。

**验证（离线、零设备）**：

- `tests/ao_shaping/scripts/test_issue238_gray_phase_probes.py` + `test_issue238_power_reference_20261120.py` + `test_issue247_geometry_capture_20261120.py` → `57 passed`，EXIT=0。
- `tests/ao_shaping/scripts/test_issue238_period_sweep_window.py`（新增）→ `7 passed`，EXIT=0。
- dry-run：`run_issue238_period_sweep_20261119.py --run-dir <Temp> --periods-px 32 48 --period-order palindrome --frames-per-level 20 --power-reference abba --power-reference-frames 4 --dark-frames 20` → EXIT=0，`DIR_CREATED=False`，子段 argv 含 `--dark-frames 20`。

**仍未闭合（B1 硬前提）**：

- **照明关断 hook 未接线**：`_run(dark_capture_hook=...)` 是留给硬件车道的 seam；当前无任何调用方提供它，因此 CLI/`--execute` 下 `--dark-frames N>0` 一律 fail-fast。B1 真采前须由一个硬件车道实现并把 hook 传入 `_run`（或补一个真实的照明关断 API）。
- **灰度层集合冻结**：dry-run 已能打印实际灰度层集合（binary 模式；run4 实测 48 层，含 864–1023 加密带），B1 前须在预注册里写死层数。

> 暗场是 §6.6 永久报告门槛的硬前提；**代码支持 + wrapper 透传已完成，但照明关断 hook 未接线前 B1 仍不能真采遮光帧**。

### 4.3 实验采集 B1（冻结参数）

| 项 | 值 |
|---|---|
| 脚本 | `scripts/experiments/run_issue238_period_sweep_20261119.py`（B0 扩展后） |
| `--run-dir` | `data/debug/hw_window_<date>c/issue238_formal_g2pi/`（唯一新目录） |
| `--periods-px` | `32 48 64` |
| `--period-order` | `palindrome` |
| `--frames-per-level` | `20` |
| `--power-reference` | `abba`，`--power-reference-frames 4` |
| `--dark-frames` | `20`（B0 新增） |
| 冷却 | 段间 ≥200 s（窗口脚本负责） |

```powershell
.venv\Scripts\python.exe scripts/experiments/run_issue238_period_sweep_20261119.py `
  --run-dir data/debug/hw_window_<date>c/issue238_formal_g2pi `
  --periods-px 32 48 64 --period-order palindrome `
  --frames-per-level 20 --power-reference abba --power-reference-frames 4 `
  --dark-frames 20 `
  --wait-until-idle 900 --execute
```

> ⚠️ 当前 `--execute` + `--dark-frames 20` 会在打开驱动前 fail-fast（`dark_abort.json`），因照明关断 hook 未接线（§4.2）。上表命令是**接线后的目标口径**；接线前只能 dry-run（不建目录、不碰设备）。

### 4.4 分析口径与永久报告门槛

沿用 `issue238_binary_gray_capture_plan_20261119.md` §6.6：

1. 输入完整性：帧数、shape/dtype、周期/灰度标签、曝光、饱和记录、扫序；暗场与 ABBA 参考齐备。
2. 固定 ROI/背景规则，逐帧定位 0/±1 级，提取二维轮廓 + 固定孔径积分 + 背景统计；不得按拟合结果删帧/改 ROI/挑灰度。
3. 嵌套模型 M0（共享轮廓）/M1（period 相关低维项），显式含帧级乘性漂移与加性背景；复杂度不构成证据。
4. 按连续时间/灰度块做留块预测与 block bootstrap；不得按帧随机拆分。
5. 只有**输入完整 + 留块预测稳定改善 + 背景/ROI/漂移敏感性下参数不贴边且方向一致**，才写永久报告（放 `report/shape_trials/`，在 `provenance.py::REPORTS` 登记并跑 provenance sync/check + 对应测试）。
6. 未满足则结论记为"当前数据无法区分机制"，**不发布新 g₂π**。

### 4.5 不做什么

- 不与 run4 的 g₂π 直接合并（口径不同）。
- 不在脚本内计算/声明新 g₂π（脚本只产描述量）。
- 不改 LUT、不推进逐像元 LUT/串扰核（cell=128/gap=64 不足以判决串扰）。

## 5. 实验 C：#234/#236 双门修复（离线，不实验）

**现状（C0 离线复核已完成，2026-11-20）**：双门均有效 fail；离线复跑与归档逐字一致（只读归档 npz，零设备，两脚本 exit 1）。

### 5.1 强度门（`data/debug/slm_transfer/p3_gate_20261120/strength_gate.json`）

- 判据 `STRENGTH_GATE_DELTA=0.20`（`strength_gate.py:31`）；指标 `masked_absolute_rel_l2`：`relL2 = ‖(pred−dc_p)−(meas−dc_t)‖_mask / max(‖(meas−dc_t)‖_mask, 1e-9)`（`strength_gate.py:159-166`），worst-pass 决定。
- 复跑：**12/12 fail**，worst `relL2=0.8848927`（`holdout_008_zernike`），median 0.7875；曝光 0.087 ms、无饱和、exposure_consistent 全 true。
- 逐样本 relL2：000/003/006 smooth=0.7878/0.8837/0.7872；001/004/007/010 bandlimited=0.7259/0.7953/0.7828/0.7649；002/005/008/011 zernike=0.8411/0.7748/0.8849/0.8056；009 smooth=0.6935（最佳）。族均值 zernike≈0.827 / smooth≈0.788 / bandlimited≈0.767。**分布窄、近似统一 ~0.79 偏置，仅轻度形状依赖** → 更像系统性 scale/registration/几何偏差，而非个别坏形状。

### 5.2 方向门（`data/debug/slm_transfer/p3_gate_20261120/direction_gate.json`）

- 判据：每对 Pearson≥0.50 且 ≥10/14 对（`transfer_upstream_qualification_plan_20261119.md:82`）；指标 `jacobian_guard.direction_agreement` 的 Pearson（`jacobian_guard.py:117`）。
- 复跑：pooled **4/14**（pass1 2/7、pass2 2/7）。逐 Zernike mode（pass1 / pass2）：
  - Z5 `0.5701✓ / 0.4933✗`
  - Z6 `0.0664✗ / 0.2041✗`
  - Z7 `−0.1833✗ / 0.2179✗`
  - Z8 `0.7970✓ / 0.7991✓`
  - Z9 `0.0704✗ / 0.0116✗`
  - Z10 `0.4444✗ / 0.3882✗`
  - Z11 `0.4807✗ / 0.5358✓`
- 只有 Z8 两 pass 都过；Z6/Z7/Z9 两 pass 都严重失败。**失败随 mode 重复，属形状/模态依赖**，不是统一偏置。

### 5.3 轴映射依赖（关键）

方向门**依赖 #247 面板↔CCD 映射**：`direction_holdout_gate_20261119.py:100-125` 把冻结的 `ramp_matrix` 传入 `FrozenDefocusParameters`，`differentiable_defocus_uniformity.py:285-286,295` 用 `inv(ramp_matrix)` 把相机采样网格映到频率网格。`ramp_matrix` 即 panel→CCD 线性映射；映射旋转/翻转错误会直接旋转预测差分图案、压低 Pearson。**映射未冻结前，方向门 fail 不能归因于模型本身。**

### 5.4 已记录的根因线索

- `agent_notes_20261118.md:44-62`：主响应轴符号失配 —— 响应 Gram PC1 符号集 `{Z5,Z8,Z10,Z11}` 与通过集（4/7）一致；残差 Zernike 项会翻转 Z6/Z7/Z9 的 PC1 符号；scale sweep 对 Pearson 不敏感，window/sigma/defocus sweep 均封顶 5/7 且失败集轮换；噪声已排除（SNR 2.26–5.04）；双门严格互斥（去残差把方向门抬到 5/7 但强度门 relL2 0.355→0.582）。
- 冻结模型：`p3_fit_20261120/frozen_parameters.json`（sha256 `380b9b87…`）；`registration_shift_ccd_px=null`（未启用）；gain/background/bias 仅在 train 拟合（预注册禁止 holdout 后拟合，`transfer_strength_gate_preregistration_20261119.md:25`）。方向门硬编码 `sigma_ccd_px=0`、`phase_sign=-1/scope="command"`，每 pass 用椭圆中心重估 origin。

### 5.5 下一步（纯离线，不实验）

1. **先 A0/A1 冻结映射**：方向门 fail 可能部分来自 `ramp_matrix` 与真实轴映射不符；映射冻结前不重判方向门（与 §2、§3.5 一致）。
2. **强度门**：`relL2≈0.79` 近统一偏置，优先查冻结前向的 registration/scale/bias/曝光归一，用现有离线帧画 bias–scale 残差图，判断整体标定偏差 vs 局部结构。
3. **重跑命令已固化**（零设备、只读归档，见 §5.6）；结果只写 Temp，不写 run 目录。
4. **门槛**：离线双门须**有裕度**通过（不是贴边）才可申请上机；否则维持暂停。本线不产生设备动作。

### 5.6 离线复跑命令（已执行，exit 1 = fail）

```powershell
.venv\Scripts\python.exe scripts\slm\evaluate_strength_gate_20261119.py `
  --dataset data\debug\slm_transfer\p3_zernike_20261120 `
  --checkpoint data\debug\slm_transfer\p3_fit_20261120\frozen_parameters.json `
  --dark-dir data\ccd_configs\ccd\FJB24112232\darks `
  --optical-path-record data\debug\optical_path_probe\p3_upstream_20261120\20261120_012500Z\optical_path_record.json `
  --data-bit-depth 8 --data-full-scale 255 --span 160 --window-step 2 --zero-radius-px 4 `
  --origin measured --radiometry train --out <Temp>\strength_gate_rerun_20261120.json

.venv\Scripts\python.exe scripts\slm\direction_holdout_gate_20261119.py `
  --runs data\debug\slm_transfer\p3_direction_pass1_20261120 data\debug\slm_transfer\p3_direction_pass2_20261120 `
  --checkpoint data\debug\slm_transfer\p3_fit_20261120\frozen_parameters.json `
  --out <Temp>\direction_gate_rerun_20261120.json --center-method ellipse --min-pearson 0.5 --min-pairs 10
```

## 6. 实验 D：#235 剩余子项（离线优先，实验可选）

- 物理 LUT、`shift_phase`、独立 pupil（瞳面）-center 未闭环。
- 先离线定义验收（与生产 `zero_order_center_by_method` 同口径），能离线的先做完。
- 若确需实验，只做最小采集（单 pupil（瞳面）-center 扫描），并入 §8 窗口，不单独开窗。

## 7. 离线守护线：#249 / #250 / #251

- #249/#250：核心修复已落地，待 owner 复核；本线只做回归，不改口径。
- #251：4 个系统性缺陷未根治；`.worktrees/issue251` 有并行 WIP，主树**不与其实例混提交**；提交逐路径 `git add`。

## 8. 设备窗口合并执行与时间预算

| 顺序 | 段 | 预计 | 前置 |
|---|---|---|---|
| 0 | 六步光路探测（`probe_optical_path_validity.py`，s0） | ~2 min | 无设备竞争 |
| 1 | A1 #247 配准 | ~5 min | A0 完成 |
| 2 | 冷却 ≥200 s | — | — |
| 3 | B1 #238 正式 g₂π（3 周期 × palindrome × 20 帧/灰度 + 暗场） | ~4 h | B0 完成 |
| 4 | 冷却 + 释放 | — | — |

- 若 B0 未就绪：窗口只做 A1（~10 min 含探测与释放）。
- 帧率按本轮数据推算（~2.1 s/帧：820 帧 / ~29 min）；B1 约 5760 sweep 帧 + 暗场/参考，**需长窗口**，建议单独安排。
- 一次窗口只开/关一次 SLM+CCD；段间由窗口脚本负责冷却。

## 9. 硬件纪律、释放与溯源

- 占用前在回复报告：设备、动作、预计释放；结束后报告：是否正常关闭、锁是否释放。
- `data/debug/.device.lock` 的 1 字节陈旧残留不阻塞（OS 字节锁无人持有）。
- 每段 `finally`：`set_grayscale(0)→cam.close→slm.close→lock.release`；写 `released.json`（`closed_normally`、`lock_released`）。
- 写 `optical_path_record.json`（`ao-optical-path-record/v1`），**补上本轮缺口**：`drive_board_id`（调用 `require_drive_board()`）、`exposure_range_ms`、`gain`、`git_commit`；释放信息写 `released.json` 而非留 null。
- CCD 全幅 Mono12，不物理开窗；需要缩小范围只在后处理裁切。
- 若只做整形相位、未加倾斜/离焦，0 级过强时可轻微过曝采集，并在记录中注明"过曝采集"。

## 10. 数据目录与 manifest 规范

```
data/debug/hw_window_<date>c/
├── optical_path_probe/s0/<UTC>/optical_path_record.json
├── issue247_axis/
│   ├── frames/*.npy            # uint16, 1944×2592 全幅
│   ├── summary.json            # 图样清单、相位 sha256、饱和计数、status
│   ├── optical_path_record.json
│   └── released.json
└── issue238_formal_g2pi/
    ├── 32/ 48/ 64/ 64_r2/ 48_r2/ 32_r2/
    │   ├── frames.jsonl        # 逐帧: gray, period_px, frame_file, role, exposure_ms, saturated, utc
    │   ├── npz/*.npz           # 原始 Mono12
    │   ├── summary.json        # power_reference / dark 段, frames_per_level=20
    │   └── release.json
    └── released.json           # 窗口级 final free
```

- `frames.jsonl` 必含：`gray`、`period_px`、`frame_file`、`frame_shape`、`frame_dtype`、`role`（sweep/dark/pre/post/bracket/drift）、`exposure_ms`、`saturated`、单调 UTC。
- 数据与清单**只读**；离线分析输出只写 Temp，不写回 run 目录。

## 11. 需用户明确批准的动作 / 未决项

| 动作 | 说明 |
|---|---|
| 实验 A1 | 需明确批准 + 新唯一 `--out`；预计 ~10 min（含探测/释放） |
| 实验 B1 | 需明确批准 + B0 脚本完成（已完成）+ **照明关断 hook 接线**（未完成，见 §4.2）；预计 ~4 h，建议单独长窗口 |
| 接线 `dark_capture_hook` | 需一个硬件车道实现照明关断并把 hook 传入 `_run`；否则 `--dark-frames N>0` 在 `--execute` 下 fail-fast |
| 写 `slm_geometry_calib.npz` | **禁止**，除非用户单独批准；A1 只产建议值 JSON |
| 改 `provenance.py` / `report/README.md` | 被并行车道持有，勿碰；永久报告须待其释放后登记 |
| 改 `data/ccd_configs/.../darks/manifest.json` | 位深混存缺陷（见 `dark_frame_record_20261119.md`），会触发 §0.3 重签，未批准不动 |

## 12. 变更控制与落盘

- 本文件为**冻结稿**；改动任一冻结参数须重新走用户确认并更新本文。
- 采集前先写/更新预注册（离线、无设备），采集后按 §10 归档、按 §4.4/§3.4 判定。
- 本轮无设备占用；下一次占用前先确认锁空闲、无竞争进程，再按 §9 报告。
- **2026-11-20 离线改动（未提交、未 push）**：`scripts/slm/issue238_gray_phase_probes.py`（暗帧 opt-in + fail-fast seam）、`scripts/experiments/run_issue238_period_sweep_20261119.py`（暗场标志透传）、`tests/ao_shaping/scripts/test_issue238_gray_phase_probes.py`、新增 `tests/ao_shaping/scripts/test_issue238_period_sweep_window.py`。按 §4.2/§5 描述执行；并行 WIP 在树中，若提交须逐路径 `add`。

---

### 关联文件

- 结果：`docs/dev/hw_window_20261120b_issue247_238_results.md`
- #238 口径：`docs/dev/issue238_binary_gray_capture_plan_20261119.md` §6.5–§6.8
- #247 历史映射：`report/shape_trials/issue247_zero_order_20261117/issue247_zero_order_20261117.md`
- 轴失配独立验证：`docs/dev/axis_mismatch_independent_capture_plan_20261118.md`
- 合并版队列：`docs/dev/hw_plan_consolidated_20261118.md`
- 采集脚本：`scripts/slm/issue247_geometry_capture_20261120.py`、`scripts/slm/issue238_gray_phase_probes.py`、`scripts/experiments/run_issue238_period_sweep_20261119.py`
- 光路标定：`libs/skills/ao-optical-path-calibration/SKILL.md`
- 硬件纪律：`docs/dev/hardware_discipline.md`
