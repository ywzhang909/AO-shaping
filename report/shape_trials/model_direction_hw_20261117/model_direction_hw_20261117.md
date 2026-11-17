# 冻结模型方向实测诊断（2026-11-17，14 帧）

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`scripts/slm/square_1000_model_diagnostic_20261030.py`](../../../scripts/slm/square_1000_model_diagnostic_20261030.py)
> **数据/关联脚本**: [`scripts/slm/analyze_square1000_model_diagnostic_20261030.py`](../../../scripts/slm/analyze_square1000_model_diagnostic_20261030.py)
> **数据/关联脚本**: [`scripts/slm/direction_peak_centred_20261117.py`](../../../scripts/slm/direction_peak_centred_20261117.py)
> **数据/关联脚本**: [`scripts/slm/issue235_knife_baseline_fit_20261117.py`](../../../scripts/slm/issue235_knife_baseline_fit_20261117.py)
> **数据/关联脚本**: [`scripts/slm/issue235_beam_centre_differential_20261117.py`](../../../scripts/slm/issue235_beam_centre_differential_20261117.py)
> **数据/关联脚本**: [`scripts/slm/issue235_panel_gray_inversion_20261117.py`](../../../scripts/slm/issue235_panel_gray_inversion_20261117.py)
> **运行环境**: 离线+硬件（Santec SLM-200 #1 + Daheng MER2-507-23GM，2026-11-17 上机采集；报告本体不占用设备）
> **说明**: 冻结模型 14 帧方向诊断（data/debug/square1000_model_diag_20261117_140741）、窗口对照（win0/win163）、M3 以实测峰为中心复核（m3_peak_centred_20261117.json）、刀口基线拟合（issue235_knife_baseline_fit_20261117.json）、差分刀口（diff_20261117_170647_v2/summary.json）与 panel_gray 反解前向一致性（issue235_panel_gray_inversion_20261117.json）、实测 panel 中心回灌的前向修正（m3_peak_centred_centerfix_20261117.json）。人工撰写。
<!-- provenance:end -->

数据：`data/debug/square1000_model_diag_20261117_140741`（SLM #1 + Daheng CCD，base 曝光 0.1 ms，探针幅度 0.03 rad，tilt 周期 64 panel px）。
设备：14:07–14:09 占用，14:09:05 释放；SLM 与 CCD 清洁关闭（`released.json` 双 true）。
采集：`scripts/slm/square_1000_model_diagnostic_20261030.py --base-exposure-ms 0.1 --wait-until-idle 300 --execute`。
离线分析：`scripts/slm/analyze_square1000_model_diagnostic_20261030.py --session data/debug/square1000_model_diag_20261117_140741` → `diagnostic_analysis.json`（已复制到本目录）。

## 量化结论

- 测量重复性：锚点重复 cosine 1.0000，差分 rms ≈ 7 counts，repeat_std ≈ 0.84 counts。测量噪声可排除。
- 锚点绝对拟合：gain 13.59，correlation −0.160，relative_l2 0.199。
- N4 离焦：实测响应 rms 3.68 counts，预测 rms 4.4e-5，cosine −0.0075。
- RX/RY 奇数响应：实测 rms 约 3.3–3.6 counts，预测 cosine ≈ 0.001–0.002。
- 光斑形态（锚点帧，`capture_0000.npz`）：峰值 1784 counts 位于 (row,col)=(1107,1225)，超过半高的像素仅 199 个，±50 px 内功率 1.1%。
- 冻结模型预测（`FrozenDefocusForward`，`load_seed_contract(SEED_RUN, CHECKPOINT, seed_key=SEED_KEY)`）为 1000×1000 像素方形平顶，填充 97%，CV ≈ 0.069。测量帧在该方形 bbox（rows 436:1448, cols 1416:2416）内的功率占比仅 19.9%，方形内外平均计数 15.23 对 15.46，没有方形结构。

## 对三个问题的回答

1. 背景误差能否单独解释方向失败？不能。测量重复性极好，本底与均匀化项不足以把锚点相关从 +1 推到 −0.16。
2. 非线性误差能否单独解释？不能。探针为 0.03 rad 小幅线性区，N4/RX/RY 的预测响应本身接近零（N4 预测 rms 4.4e-5），而实测有 3–4 counts。缺的是响应本身，不是非线性项。
3. PSF/defocus/照明包络是否仍主导？在本窗口下是主导的：测量是衍射受限的单点聚焦，模型是方形平顶。离焦与照明包络的拟合无法把平顶变成点，问题更像是窗口/中心注册错误或物理预测模型的远场形态错误。

## 注册前提（需先核实）

- 冻结 checkpoint 的 `origin_xy` = (x 1265.7, y 941.1)，现场几何 `data/slm_geometry_calib.npz` 中心 = (x 1224.4, y 1104.8)，差约 (dy,dx)=(+163.7,−41.7) px。
- 本次测量峰值 (1107,1225) 与现场中心吻合，而模型窗口（rows 300:1584, cols 1300:2536，`camera_xy`）按归档中心构造。因此本次 "方向失败" 可能部分来自窗口错位，不能直接当作模型的最终判决。
- 尝试过的整片传感器数据驱动配准脚本 `scripts/slm/direction_reregistration_20261117.py` 未得到有效结果（high-pass 相关 −inf，探针键名错误），不作为证据，也未提交。

## 与历史结论的矛盾

- `report/shape_trials/direction_diagnosis_20261112/direction_diagnosis_20261112.md` 的 M3（离焦+照明包络 median 0.616/0.696/0.657 过 ≥0.50 门禁）与本次结论冲突。两者口径不同（归档帧与旧窗口 vs 新鲜硬件帧，探针幅度不同）。本报告不推翻历史，待对账。

## 刀口基座根因与差分刀口（2026-11-17，实验）

- **刀口法失效的根因**：`data/debug/knife_block_20261111*` 显示，遮挡图案在 0 级窗口内仍保留 0.41–0.43（checker 8–48 px）、0.59（blaze）、0.61–0.65（随机相位）。`find_beam_on_slm` 的刀口曲线因此必然带约 40–60% 基座，50% 交点不是光束中心。此前基于单刀口的测量与 `--pupil-center-xy` 结论均不可用。
- **坐标系修正**：`data/slm_geometry_calib.npz` 的 `center`（1104.84, 1224.42）是 **CCD 坐标**（与 CCD 零级 argmax (1107, 1227) 一致，`beam_center` 字段为 [0,0] 从未写入）；刀口中心是 **panel 坐标**。二者不可直接比较，之前把它们当同一量对比是错误的。
- **基线猜想撤回**：自由基线 erf 拟合与 CDF 分位一致，基线偏置不解释偏移。
- **差分刀口方法**：`scripts/slm/issue235_beam_centre_differential_20261117.py`。每个分割位置 s 同时测"清左半边"(A) 与"清右半边"(B)，`D(s)=A−B` 抵消基座，`D(s)=amp·erf((s−c)/(√2σ))`。
- **实验**：SLM #1（驱动板 `22030108`，`serial_source=driveboard`）+ Daheng CCD cam_id 0，0.05 ms，峰 187/255 未饱和。17:06:47–17:09:50 占用，`release.json` 三项 true，正常关闭。数据：`data/debug/issue235_beam_center/diff_20261117_170647_v2/summary.json`。
- **结果（panel 坐标，cy/cx 约定）**：

| 轴 | 窗口 | 过零 | erf 拟合中心 | σ | 残差/幅度 | 自由基线中心 | 基座占比 |
|---|---|---|---|---|---|---|---|
| x | 6× | 880.9 | 887.9 | 148.4 | 0.026 | 892.4 | 0.50 |
| x | 12× | 886.9 | 890.8 | 148.5 | 0.028 | 887.1 | 0.54 |
| x | 24× | 882.8 | 893.8 | 138.1 | 0.047 | 886.4 | 0.62 |
| y | 6× | 589.2 | 593.7 | 161.9 | 0.038 | 590.4 | 0.52 |
| y | 12× | 583.3 | 587.1 | 145.4 | 0.038 | 576.5 | 0.54 |
| y | 24× | 587.3 | 585.6 | 155.7 | 0.040 | 571.1 | 0.63 |

- 各窗口符号翻转仅 1 次（`sign_changes=1`）；6×/12×/24× 中心离散 ≤13 px，结论稳健。
- **采用值**：panel 光束中心 ≈ (x 889, y 590)。

## 改进猜想效果汇总

| 猜想 | 结论 | 依据 |
|---|---|---|
| 背景误差解释方向失败 | 不成立 | 锚点重复 cosine 1.0000，差分 rms ≈ 7 counts |
| 非线性误差解释方向失败 | 不成立 | N4 预测 rms 4.4e-5 vs 实测 3.68 counts |
| 基线偏置致刀口中心偏 | 不成立 | erf 自由基线拟合与 CDF 分位一致 |
| 窗口错位（模型 origin vs 测量峰） | 部分成立，非主因 | 换到实测峰中心后 M3 median 仍 ≈0.025，0/3 ≥0.50 |
| 刀口单曲线可测光束中心 | 不成立 | checker 保留 0.41–0.65，需差分抵消 |
| PSF/离焦/照明包络主导 | 本窗口成立（主因） | 测量为点状聚焦（±50 px 内 ~1% 功率），模型为 97% 填充方形平顶 |

结论：方向失败主因指向物理预测模型远场形态（PSF/离焦/照明包络）与绝对强度表征，而非背景、非线性或中心窗口错位。M3 不能作为模型方向正确的证据。

## 状态

- 未训练新 checkpoint，未做锚定闭环，未评论或关闭 issue #229 / #233 / #234。
- 下一步候选：在现场中心 (1224.4, 1104.8) 重建窗口后重算本诊断；或用本次 14 帧重拟合离焦与照明包络；或设计最小设备实验分离 PSF 与非线性贡献。

## M3 方向门禁复核：以实测峰为中心的 128×128 裁剪（2026-11-17，离线）

- 脚本：`scripts/slm/direction_peak_centred_20261117.py`（离线，未碰硬件，未取 DeviceLock）。结果：`m3_peak_centred_20261117.json`。
- 裁剪中心：实测峰 native (row 1105, col 1224)，即 xy = (1224, 1105)，三模型统一使用，不再用各自拟合 origin。
- 口径：`jacobian_guard.direction_agreement`，Pearson，rate = 原始帧均值 / exposure_ms，phase_sign = −1，scope = "command"；每对以 minus 帧为锚、plus 帧为候选。
- 与历史 M3 通过结果不一致：历史 2026-11-12 为 10/14 单对 ≥ 0.50、median 0.616–0.696；本次两个会话的 M3 三对中 **0/3 ≥ 0.50**，median 约 0.025。
- 偏离历史口径：历史 128 px 网格无法接收 640×640 的 model_phase，故 `grid_size` 取检查点的 640；会话只有 N4/RX/RY 三对（每会话 14 帧），无历史 pass1/pass2 × Z5–Z11 七模。

| 会话 | 模型 | N4 | RX | RY | median | 单对 ≥0.50 |
|---|---|---|---|---|---|---|
| win0 | M0 | 0.002 | 0.001 | −0.011 | 0.001 | 0/3 |
| win0 | M1_defocus | −0.001 | 0.000 | −0.001 | −0.001 | 0/3 |
| win0 | M3_illumination | nan | nan | 0.026 | 0.026 | 0/3 |
| win163 | M0 | −0.012 | −0.001 | −0.010 | −0.010 | 0/3 |
| win163 | M1_defocus | 0.004 | 0.001 | −0.003 | 0.001 | 0/3 |
| win163 | M3_illumination | nan | nan | 0.025 | 0.025 | 0/3 |

- M3 的 N4/RX 为 `insufficient_signal`（nan），仅 RY 有值；median 与计数基于极少样本，不能作为门禁判决。
- 结论：在实测峰为中心的裁剪下，历史 M3 方向通过**未复现**。M3 的 origin 也被固定在峰上，与历史 fitted-origin 口径不同，因此差异不能单独归因于照明包络。
- 已补：`diagnostic_analysis_win0.json` / `diagnostic_analysis_win163.json` 由 `scripts/slm/analyze_square1000_model_diagnostic_20261030.py --session` 对两个窗口会话重跑生成，存档在本目录（见下节）。
- 冻结检查点 `model640/frozen_parameters.json` 未单独评估（本次只用历史 fit JSON 的 M0/M1/M3 参数）。
- 溯源：本节手写追加；本报告已在 `scripts/_common/provenance.py::REPORTS` 注册，头部由 `sync_report_provenance.py` 生成。

## panel_gray 反解与前向输入一致性（离线，issue #235 §2(1)(a)(b)）

脚本 scripts/slm/issue235_panel_gray_inversion_20261117.py，输入 data/debug/square1000_model_diag_20261117_140741（不占设备）。输出 issue235_panel_gray_inversion_20261117.json。

- 编码校验：floor(mod(phase,2π)/2π·1023) 经 to_panel 展开后与存档 panel_gray 逐像素一致（14 帧全部 mismatch 0）。
- 反解残差：按 1023 标度反解，相对请求相位 wrapped rms 3.55e-3 rad（即量化）；按物理 993 标度反解为 0.107 rad（相位尺度压缩约 3%，0.03 rad 探针下不可忽略）。
- 响应对比（drift-corrected 实测 vs 预测奇响应 cosine）：N4 requested -0.0067 / actual_1023 -0.0052 / actual_993 -0.0067；RX 0.0025 / 0.0025 / 0.0031；RY 0.0014 / 0.0014 / 0.0020。

结论：前向输入换成 panel_gray 反解的实际相位后，方向 cosine 仍约为 0，增益仅数值变化，不改变方向结论。输入相位不一致不能解释方向失败；主因仍在远场形态模型（PSF/defocus/照明包络）。

### 两个窗口会话的 analyzer 存档

- 输入：`data/debug/square1000_model_diag_win0_20261117`（无窗）与 `..._win163_20261117`（±163 px 窗），各 14 帧、0.1 ms、无优化。
- null 对照：两会话 5 对锚点 cosine 均 1.0000，差分 rms 6.72–7.98 counts（重复性）。
- 方向响应（drift-corrected cosine）：win0 为 N4 −0.0018 / RX 0.0018 / RY −0.0003；win163 为 N4 0.0009 / RX 约 0.000 / RY 约 0.000。旁路最优变换（transpose / flip / 180°）与最优平移（−20…+12 px）都无法把 cosine 提到 0.02 以上。
- 偶响应主导：实测 even/odd rms 比 1.28–1.35，模型同量比仅 0.011–0.016。探针的奇响应被约 40 倍强的偶分量淹没，这是"模型方向对了也测不出来"的直接量化。
- 结论：方向失败不是窗口选择造成的；即使允许任意坐标变换与平移对齐，模型仍与实测无关。

## 前向修正：用实测 panel 中心替换照明包络中心（离线）

差分刀口给出 panel 光束中心 (x 889, y 590)、sigma 138–162 px。把它写进前向的照明包络中心重跑 M3 门禁（脚本 direction_peak_centred_20261117.py 新增 `--amplitude-center-xy`，结果 `m3_peak_centred_centerfix_20261117.json`）：

| 会话 | 模型 | 原 median | 换中心后 median | 单对 ≥0.50 |
|---|---|---|---|---|
| win0 | M0 | 0.001 | 0.003 | 0/3 |
| win0 | M1_defocus | −0.001 | 0.001 | 0/3 |
| win0 | M3_illumination | 0.026 | 0.035 | 0/3 |
| win163 | M0 | −0.010 | −0.011 | 0/3 |
| win163 | M1_defocus | 0.001 | −0.002 | 0/3 |
| win163 | M3_illumination | 0.025 | 0.042 | 0/3 |

- 照明中心偏移只能把 M3 的 median 从 0.025–0.026 抬到 0.035–0.042，离 0.50 差一个数量级；照明包络位置不是方向失败的主因。
- 顺带发现一处模型内部不一致（未修）：冻结检查点 `model640/frozen_parameters.json` 的 amplitude_center 为 (x 965.0, y 909.9)，同一模型的 fit JSON 为 (x 1018.3, y 581.0)，实测为 (x 889, y 590)。检查点那一路的 y 比另外两者高约 320–330 px，建议后续按轴序/来源核对；本次只验证换中心不改变方向结论，未据此改检查点。
