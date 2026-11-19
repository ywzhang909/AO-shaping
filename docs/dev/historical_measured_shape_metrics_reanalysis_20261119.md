# 历史实测目标形状指标重算与判定修正（2026-11-19）

<!-- provenance:start -->
> **生成脚本**: 人工撰写；离线重算脚本位于 `C:/Users/zhangh/AppData/Local/Temp/opencode/`（非仓库内，见 §6）
> **数据/关联脚本**: [`src/ao_shaping/tools/slm/transfer/protocol.py`](../../src/ao_shaping/tools/slm/transfer/protocol.py)（`target_intensity_rmse`，冻结指标实现）
> **数据/关联脚本**: [`src/ao_shaping/algorithm/signal_processing/multishape_objective.py`](../../src/ao_shaping/algorithm/signal_processing/multishape_objective.py)（`image_metrics`）
> **数据/关联脚本**: [`report/shape_trials/physical_refine/target_shape_ccd_acceptance_plan_20261118.md`](../../report/shape_trials/physical_refine/target_shape_ccd_acceptance_plan_20261118.md)（指标契约与预注册规则）
> **数据/关联脚本**: [`issue_validity_verdict_20261119.md`](issue_validity_verdict_20261119.md)（本文件修正其 §2/§3 的失实结论）
> **运行环境**: 纯离线重算；只读历史 `.npz`，未连接/占用 SLM、CCD、激光。本轮无设备会话，`device_lock.json` 不存在。
> **说明**: 用生产函数对历史归档做 measured-vs-target 离线重算，核查 `issue_validity_verdict_20261119.md` 中「缺 measured-vs-target 配对帧 / 独立 CCD 目标形状对比不存在」的说法是否成立，并界定这些证据能/不能支撑的判定。**不设定阈值、不宣称验收通过、不升默认、不合并。**
<!-- provenance:end -->

## 1. 结论摘要（先读）

1. **旧结论失实，必须更正**：`issue_validity_verdict_20261119.md` §3 与 §2 称「不是没有数据，而是缺 measured-vs-target 的配对目标形状帧」「独立 CCD 目标形状对比不存在」。离线盘点找到 **325 个含完整数组（`raw_frames` / `target` / `core` / `camera_xy`）的历史归档目录、共 4222 个文件**，配对帧一直存在。用生产函数可直接重算实测指标。
2. **存在两组性质不同的历史证据**：
   - **20261024 系（anchored / no-GS CV）**：8-bit、无饱和，但 seed→final 实测指标几乎不动且方向不一致（`rel_l2` 甚至变差）。**不支持算法改善**。
   - **s1200 common_replay（20261028）**：12-bit、同一 replay 条件下 seed 与多组 final 相位各 3 档曝光配对比较。非饱和的 9 对里有 8 对 `rel_l2` 下降 + `energy_fraction` 上升，且 final 越优化越好。**这是真实的、相对 seed 的实测整形改善证据**。
3. **但不能据此宣称「算法有效 / 验收通过」**：
   - 绝对保真度仍差：`rel_l2` 约 0.5–1.4、`energy_fraction` 约 0.2–0.65，离任何合理通过线很远；
   - **验收阈值从未预注册**（`target_shape_ccd_acceptance_plan_20261118.md` §2 明确「阈值待现场预注册」），历史 replay 还用了 `floor=0` 而契约要求用当次冻结背景 floor；
   - s1200 是**历史存档相机的重放**，不是新鲜独立闭环采集；个别曝光有数百个饱和像素，须剔除；
   - 证据只说明「端到端整形相位优化相对初始相位有改善」，**不指向任何单一候选算法**（#234/#235/#236/#238/#247 各自的子项收益仍未被这些数据验证）。
4. **判定**：5 个开放 issue 保持 OPEN；把「无实测数据、唯一路径是实验取数」的理由改为「有历史实测配对数据，显示相对改善但不足以通过预注册验收；绝对保真度远未达标；需冻结阈值与独立采集后再评分」。

## 2. 方法与契约口径

- 对每份 `.npz`：`mean_frame = raw_frames.astype(float64).mean(0)`，按 `camera_xy` 采到与 `target` 同 grid 的窗口 `sample`。
- 用 `target_intensity_rmse(sample, target)` 得 `rel_l2 = ||m-t||/||t||` 与 `energy_fraction`（整窗各自归一化后相减）。
- 用 `image_metrics(sample, ShapeObjective(target, cv_only=True))` 得 `cv` / `ee_roi` / `target_pearson`。
- 饱和判据：`mean_frame >= full_scale * 250/255`，其中 `full_scale` 按 dtype 取值（uint16→4095 生效 12-bit；uint8→255）。
- 逐帧 `rel_l2` 的极差作为**测量噪声地板**（同一相位、同一曝光的重复帧离散度），用于判断 seed→final 变化是否超过噪声。

所有数字可用临时脚本 `C:/Users/zhangh/AppData/Local/Temp/opencode/hist_traj.py` 复算，结果在 `hist_traj.json`。

## 3. 历史实测证据

### 3.1 20261024 系：无饱和，但 seed→final 无有效改善

| 归档 / 形状 | seed rel_l2 → final | seed efrac → final | 结论 |
|---|---|---|---|
| anchored / square | 0.5561 → **0.5627** | 0.5248 → **0.5131** | 两项均变差 |
| anchored / rectangle | 0.5064 → 0.5035 | 0.6370 → **0.6296** | rel_l2 微降、efrac 变差 |
| anchored / triangle | 0.6439 → 0.6437 | 0.4024 → **0.3980** | 基本不动、efrac 变差 |
| no_gs_cv / square | 0.5602 → 0.5582 | — | 变化 < 噪声 |
| no_gs_cv / triangle | 0.6283 → **0.6367** | — | 变差 |

逐帧 `rel_l2` 极差仅约 0.0001–0.0011；anchored square 的 +0.0066 虽略超噪声，但方向是**变差**，且与 efrac 变差一致。这组数据**明确不支持**「优化改善了实测形状保真度」。全部 36 份 anchored 记录饱和像素为 0，数据质量本身干净。

### 3.2 s1200 common_replay：受控 seed→final，相对改善显著

`data/debug/slm_transfer/multishape_scale_20261028/s1200/common_replay/`：48 条 = 16 标签 × 3 档曝光，同一重放条件下比较 seed 相位与各 final 相位。`manifest.json` 明确标注 `<shape>_seed` / `..._final`。下表为**同档曝光配对**（索引 0/1/2 为三档固定曝光，由 manifest 顺序一一对应），已标出饱和：

| 形状 | 曝光档 | seed rel_l2 → transport final | seed efrac → transport final | 饱和(seed/final) |
|---|---|---|---|---|
| square | 0 | 0.978 → **0.802** | 0.428 → **0.498** | 0/0 |
| square | 1 | 1.250 → **0.938** | 0.418 → **0.531** | 0/0 |
| square | 2 | 1.430 → **1.037** | 0.412 → **0.550** | 0/0 |
| triangle | 0 | 1.072 → **0.939** | 0.168 → **0.214** | 0/0 |
| triangle | 1 | 1.282 → **1.019** | 0.180 → **0.253** | 0/0 |
| triangle | 2 | 1.394 → 1.070 | 0.187 → 0.276 | **671/259（剔除）** |
| star | 0 | 0.877 → **0.826** | 0.229 → **0.266** | 0/0 |
| star | 1 | 0.892 → **0.805** | 0.275 → **0.334** | 0/0 |
| star | 2 | 0.914 → **0.805** | 0.296 → **0.370** | 0/0 |
| circle | 0 | 0.888 → **0.771** | 0.358 → **0.433** | 0/0 |
| circle | 1 | 0.992 → **0.813** | 0.365 → **0.480** | 0/0 |
| circle | 2 | 1.062 → **0.807** | 0.369 → 0.349（efrac 例外） | 0/0 |

- 非饱和 11 对中 `rel_l2` **全部下降**，`energy_fraction` 上升 10 对（唯一例外 circle 曝光 2 的 efrac 0.369→0.349，但同对 rel_l2 仍 improvement）。
- 优化更强的方法更好：`mono12_cv_square_final` rel_l2 0.703/0.771/0.829、`mono12_continue_square_final` 0.695/0.755/0.812，均优于 `transport_cv_square_final` 的 0.802/0.938/1.037。改善随优化程度单调，符合「优化真的起作用」而非采集漂移。
- 量级（Δrel_l2 ≈ 0.05–0.39）远超 20261024 的逐帧噪声（~0.001），是真实的相对改善。

### 3.3 绝对保真度仍差，且残差不是对准/背景伪影

对若干 final 记录做了三项排除性检查（均为离线、只读存档）：

- **能量分解**：square final 目标内能量 51.3%、core 44.8%、零级盘(r30) 20.8%、其余 47.8%；三角形目标内 39.8%、零级盘 29.1%。峰位在目标内但偏离中心 20–27 px；缺能**不能**只用零级解释。
- **配准搜索**：4 个 90° 旋转 × ±30 px 平移内，最优配准仅解释 **0.0–0.5%** 的 rel_l2 距离 ⇒ 残差是强度分布本身，不是错位。
- **背景 floor 敏感性**：把 floor 从历史 replay 的 0 改为实测候选（halo_mean / p10 / offtarget_median / smoothed_p5），rel_l2 在 0.47–0.80 之间变动，efrac 最高约 0.75；结论（离目标仍远）对 floor 稳健。注意契约要求用**当次冻结 floor**，历史 replay 的 `floor=0` 口径不合规，故这些数值只作诊断、不作验收评分。
- **窗口截断检查**：采样窗仅含全帧 4–5% 能量，但窗外最大像素仅 16–20 counts（窗内 90–171），全局峰在窗内 ⇒ 窗没有漏掉被整形的光斑。

## 4. 能支撑 / 不能支撑的判定

**能支撑**：
- 历史配对实测数据存在且可重算（推翻旧结论）；
- 端到端目标形状相位优化相对初始相位，在受控重放中带来了**相对**的实测保真度提升（s1200）；
- 改善随优化方法强度单调，排除纯噪声解释。

**不能支撑**：
- 任何形式的「验收通过」：无预注册阈值、绝对 rel_l2 仍 0.5–1.4、历史 floor 口径不合规、部分帧饱和、非新鲜独立采集；
- 任何**单一** issue 子项的算法收益（#234 zernike_block、#235 输入相位一致性、#236 拟合损失、#238 LUT/串扰、#247 零级可微校正均未被这些聚合指标单独验证）；
- 升默认 / 合并。

## 5. 对 5 个开放 issue 的处理

| issue | 本轮证据是否改变判定 | 状态 |
|---|---|---|
| #234 | 否。s1200 改善来自集成优化，不特指 zernike_block 三档微调；该候选仍无双门 checkpoint、无 CV≤0.10 实测。 | OPEN |
| #235 | 否。无针对输入相位一致性 / box centroid 的单独实测消融。 | OPEN |
| #236 | 否。拟合损失重构未实施，无对应消融。 | OPEN |
| #238 | 否。逐像元 LUT / 串扰核未实施；灰度简并未解。 | OPEN |
| #247 | 否。零级可微模板 / 统一观测校正未实现；回测 r² 仍含 0。 | OPEN |

统一修正点：把 11-19 置顶评论里「下一步唯一前进路径：实验取配对 CCD 实测」「独立 CCD 目标形状对比不存在」改为事实陈述——历史配对数据存在并显示相对改善，但不足以通过预注册验收；仍需冻结阈值与做新鲜独立采集后再评分。

## 6. 复算入口与留痕

- 临时脚本（不在仓库，避免污染工作区）：`hist_traj.py`（轨迹+噪声）、`analyze_hist_metrics.py`（初筛）、`energy_decompose.py`、`align_check.py`、`floor_sweep.py`，均在 `C:/Users/zhangh/AppData/Local/Temp/opencode/`。
- 结果 JSON：`hist_focus.json`、`hist_traj.json`（同目录）。
- 关键只读数据：`data/debug/slm_transfer/live_no_gs_cv_anchored_20261024/`、`live_no_gs_cv_20261024/`、`multishape_scale_20261028/s1200/common_replay/`、`transfer_tri_star_20261027/anchored/`。
- 硬件占用：无。所有操作为离线文件读取与 NumPy 计算。
