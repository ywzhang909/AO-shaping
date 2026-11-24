# Issue #247：波形族归因窗口 v2 实测报告（2026-11-23 / 采集于 2026-11-24）

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`scripts/slm/issue247_geometry_capture_20261120.py`](../../../scripts/slm/issue247_geometry_capture_20261120.py)
> **数据/关联脚本**: [`scripts/slm/issue247_waveform_probe_solve_20261122.py`](../../../scripts/slm/issue247_waveform_probe_solve_20261122.py)
> **数据/关联脚本**: [`scripts/slm/probe_optical_path_validity.py`](../../../scripts/slm/probe_optical_path_validity.py)
> **运行环境**: 离线+硬件（Santec SLM-200 #1 1064 nm + Daheng MER2-507-23GM cam 0，2026-11-24 采集；报告本体不占用设备）
> **说明**: Issue #247 波形族归因窗口 v2：23 图样 × 4 帧 = 92 帧全幅 Mono12 采集，补齐三族 ± 对照与预注册 held-out 方向。G1 波形依赖门 fail（x 族极差 89.84° pass、y 族 1.18° fail）、G2 单族异常门 fail（无单一异常族）、G3 轴分离门 fail（grating 0.35° / tilt 0.74° 轴路由失败，blaze_patch 89.997° pass）、G4 符号一致性门 fail（blaze_patch/y 偏离 164.01° 超 5° 容差）。attribution_label = not_assessable，held-out 轴向正确 0/10，映射仍歧义、不冻结。旧 6 值 verdict tilt_specific 仅兼容保存，不作为本窗口结论。人工撰写。
<!-- provenance:end -->

**结论先行**：本窗口按预注册 `docs/dev/hw_window_20261123_issue247_waveform_attribution_prereg.md`（采前冻结）执行，92 帧采集数据有效性门全部通过，但**归因门 G1/G2/G3/G4 均 fail**，`attribution_label = not_assessable`，**无单一异常族可归因**。实测显示轴路由**随波形族与位置变化**：x 族 grating/tilt 落 CCD 列轴、blaze_patch 落 CCD 行轴，y 族三族几乎全部落 CCD 列轴。因此 #247 的 panel↔CCD 映射**仍歧义、不冻结**，不得写任何「归因规则可外推到未见方向」的结论。

- 数据：`data/debug/hw_window_20261123_issue247_waveform_attribution`（92 帧 + `plan.json` / `summary.json` / `released.json` / `waveform_probe_analysis.json`）
- 预注册：`docs/dev/hw_window_20261123_issue247_waveform_attribution_prereg.md`（唯一判据来源）
- 设备：Santec SLM-200 #1（1064 nm）+ Daheng MER2-507-23GM cam 0 + 1064 nm 激光；S0 光路探测 `verdict = ok`
- 旧 `verdict`：`tilt_specific` —— **兼容保存，不是本窗口结论，禁止引用**

## 1. 任务项

| 项 | 内容 |
|---|---|
| 目标 | 按预注册 v2 窗口补齐 #247 波形族归因：三族（grating / tilt / blaze_patch）± 对照 + 预注册 held-out 方向 |
| 前置门 | `probe_optical_path_validity.py --session issue247_attribution_20261123 --wait-until-idle 1800`，`verdict` 必须 `ok`，且 0.04 ms 落在可用曝光区间 |
| 采集 | `issue247_geometry_capture_20261120.py --plan attribution --execute`，23 图样 × 4 帧，单次 open + 一次 close |
| 分析 | `issue247_waveform_probe_solve_20261122.py --dir <capture_dir>`，G1–G4 归因门 + held-out 独立验证 |
| 判据冻结 | 曝光 0.04 ms、settle 0.8 s、饱和门 0.9×满量程、Mono12 全幅，均在下发任何帧之前写定 |
| 不做 | 不 push、不改 issue 状态、不写正式 `issue247_axis_map.json`、不发布 LUT/效率标定、不声称 #234/#236 有增益 |

## 2. 指标参数与采集记录

| 参数 | 值 |
|---|---|
| 曝光 | 0.04 ms（S0 可用窗口 0.02 / 0.04 / 0.06 / 0.1 ms 内） |
| settle | 0.8 s |
| 帧数 | 23 图样 × 4 帧 = 92 帧 |
| 位深 / 满量程 | Mono12 / 4095 counts（饱和门 3685.5 counts = 0.9×满量程） |
| SLM | Santec SLM-200 #1，1064 nm，panel 1200×1920，`two_pi_grayscale = 993`（execute 时从 SLM 实读） |
| CCD | Daheng MER2-507-23GM cam 0，12-bit，full_scale 4095，pixel 2.2 µm |
| 采集耗时 | 68.64 s |
| 采集状态 | `summary.status = ok`、`n_frames = 92`、`saturation_count = 0`、23 个 ID 全到齐 |

### 2.1 S0 前置光路探测（2026-11-24 01:11:48Z → 01:13:17Z）

| 项 | 值 |
|---|---|
| 命令 | `probe_optical_path_validity.py --session issue247_attribution_20261123 --bit-depth 12 --cam-type daheng --cam-id 0 --slm-number 1 --slm-wavelength 1064 --exposures 0.02,0.04,0.06,0.1 --defocus-amps 0,0.5,1.0,-1.0 --wait-until-idle 1800` |
| verdict | `ok`（`probe_status = ok`） |
| 可用曝光 | 0.02 / 0.04 / 0.06 / 0.1 ms |
| 噪声底 | 20.93 counts（0.02 ms 档 `noise_threshold_counts`） |
| 饱和 | 无（0.04 ms 档峰值 2383.6 counts < 3890.25 counts 门限） |
| 离焦响应 | +1 rad 离焦宽度 +4.06 px、峰值 −628.8 counts（响应显著，光路有效） |
| 记录 | `data/debug/optical_path_probe/issue247_attribution_20261123/20261124_011148Z/optical_path_record.json`，`occupancy.closed_normally = true` |

0.04 ms 落在 S0 可用曝光区间内，满足预注册前置条件，采集放行。

### 2.2 92 帧采集与设备释放（2026-11-24 01:15:36Z → 01:16:45Z）

| 项 | 值 |
|---|---|
| 命令 | `issue247_geometry_capture_20261120.py --plan attribution --execute --out data/debug/hw_window_20261123_issue247_waveform_attribution --exposure-ms 0.04 --settle-s 0.8 --slm-number 1 --wavelength-nm 1064 --cam-id 0 --wait-until-idle 1800` |
| 采集开始 / 结束 | 2026-11-24T01:15:36.708Z → 2026-11-24T01:16:45.344Z |
| 耗时 | 68.6356 s |
| SLM | 单次 open（memory 模式）→ 关闭前 `set_grayscale(0)` → 关闭；`slm_closed = true` |
| CCD | 打开 → 关闭；`camera_closed = true` |
| 锁 / 正常关闭 | `lock_released = true`、`closed_normally = true`、`status = ok`、`error = null` |
| 残留进程 | 无（关闭前 SLM 灰度置 0，无残留 bench 进程） |

`released.json` 全部释放项为 `true`，采集在飞保护与设备关闭均正常。

## 3. 数据有效性门（先判，不过则整窗 invalid）

| 门 | 判据 | 实测 | 状态 |
|---|---|---|---|
| 采集完整性 | `status == ok`、`n_frames == 92`、23 ID 全到齐 | `ok` / 92 / 全到齐 | pass |
| 饱和门 | `saturation_count == 0`、单图样峰值 < 0.9×满量程 | 0；最大峰值 1781 counts（`flat_pre`） | pass |
| flat 漂移门 | `flat_pre/flat_mid/flat_post` 两两质心差最大 ≤ 2.0 px | 最大 1.06 px（`flat_pre→flat_mid`） | pass |

数据有效性门全部通过，整窗进入归因判读。

## 4. 归因门 G1–G4 逐门实测

判据口径：角 θ(f,a) = 0 级 → ±1 级位移向量在 CCD 平面上的角度（grating 用 −1→+1 向量）；CCD 列 ≈0°、CCD 行 ≈90°。期望轴分离 88.544°（A 映射口径），容差 2°。本节按预注册 §3.2 独立判读，**旧 `gates` / `verdict` 仅为兼容保存，不是本窗口的结论**。

| 归因门 | 状态 | 实测 | 判据 |
|---|---|---|---|
| G1 波形依赖[x] | dependent（pass） | 极差 89.84°（grating=1.43°, tilt=0.82°, blaze_patch=-88.73°） | >5.00° pass / ≤2.00° fail |
| G1 波形依赖[y] | independent（fail） | 极差 1.18°（grating=1.08°, tilt=0.08°, blaze_patch=1.27°） | >5.00° pass / ≤2.00° fail |
| **G1 波形依赖（总）** | **fail** | 两轴合并 | 两轴均 pass 才 pass |
| G2 单族异常[grating] | fail | 偏离 x=45.38°, y=0.40°；另两族互差 x=89.55°, y=1.18° | 两轴偏离 >5° 且另两族 ≤2° |
| G2 单族异常[tilt] | fail | 偏离 x=45.52°, y=1.09°；另两族互差 x=89.84°, y=0.19° | 两轴偏离 >5° 且另两族 ≤2° |
| G2 单族异常[blaze_patch] | fail | 偏离 x=89.86°, y=0.69°；另两族互差 x=0.60°, y=0.99° | 两轴偏离 >5° 且另两族 ≤2° |
| **G2 单族异常（总）** | **fail** | `anomalous_family = —`（无单一异常族） | 恰一族满足才 pass |
| G3 轴分离[grating] | fail，轴路由失败 | abs(θx−θy)=0.35° | 88.544° ± 2.00° |
| G3 轴分离[tilt] | fail，轴路由失败 | abs(θx−θy)=0.74° | 88.544° ± 2.00° |
| G3 轴分离[blaze_patch] | pass | abs(θx−θy)=90.00° | 88.544° ± 2.00° |
| **G3 轴分离（总）** | **fail** | grating、tilt 轴路由失败 | 全族 pass 才 pass |
| G4 符号[tilt/x] | pass | θ(+)−θ(−)=180.13° | 180° ± 5° |
| G4 符号[tilt/y] | pass | θ(+)−θ(−)=179.81° | 180° ± 5° |
| G4 符号[blaze_patch/x] | pass | θ(+)−θ(−)=184.25° | 180° ± 5° |
| G4 符号[blaze_patch/y] | fail | θ(+)−θ(−)=164.01°（偏离 15.99°） | 180° ± 5° |
| **G4 符号一致性（总）** | **fail** | blaze_patch/y 超容差 | 全对 pass 才 pass |

**逐门小结**：G1/G2/G3/G4 **全部 fail**，未通过的门不被其它门抵消。G1 的 x 轴虽有 89.84° 的波形依赖，但 y 轴仅 1.18°，两轴合并不满足「波形间方向差」判据；G2 无一族在两轴上同时偏离另两族 >5° 且另两族互差 ≤2°，无法定位单一异常族；G3 只有 blaze_patch 复现 ≈90° 轴分离，grating 与 tilt 两轴几乎同向（0.35°/0.74°），轴路由失败；G4 仅 blaze_patch/y 一对符号不一致。

## 5. held-out 独立验证

判据口径按预注册 §3.3：角度统一用 `axis_angle_deg`，CCD 列 ≈0° / 行 ≈90°。预测规则由 §3.2 唯一决定，冻结后不回改；G1/G2 均 fail 且 G3 未全过，故 `θ̂` 记为 `not_assessable`。

| 指标 | 值 |
|---|---|
| held-out 门 | **fail**（`gate_pass = false`） |
| 预测规则 | `not_assessable` |
| 轴向正确率 | **0.0**（0/10） |
| per-family | grating 0/2、tilt 0/4、blaze_patch 0/4（均 `accuracy = 0.0`、`gate_pass = false`） |
| 轴向正确门槛 | `axis_accuracy_min = 0.9`、per-family ≥ 2/3 |

| ID | 族 | 轴 | 周期(px) | 实测轴向角(°) | 实测轴向类别 | 状态 |
|---|---|---|---|---|---|---|
| `grating_x_p192` | grating | x | 192 | 0.74 | column | not_assessable |
| `tilt_x_plus_p192` | tilt | x | 192 | 1.79 | column | not_assessable |
| `tilt_x_minus_p192` | tilt | x | 192 | -179.78 | column | not_assessable |
| `patch_x_plus_p192` | blaze_patch | x | 192 | -88.06 | row | not_assessable |
| `patch_x_minus_p192` | blaze_patch | x | 192 | — | — | not_assessable |
| `grating_y_p192` | grating | y | 192 | 1.41 | column | not_assessable |
| `tilt_y_plus_p192` | tilt | y | 192 | -1.09 | column | not_assessable |
| `tilt_y_minus_p192` | tilt | y | 192 | -179.52 | column | not_assessable |
| `patch_y_plus_p192` | blaze_patch | y | 192 | -12.18 | column | not_assessable |
| `patch_y_minus_p192` | blaze_patch | y | 192 | -167.21 | column | not_assessable |

- `patch_x_minus_p192` 的 `axis_angle_deg = None`、`lobes = []`：瓣检测未命中，**无有效角度测量**，如实记录，不计入轴向正确。
- held-out 未通过独立验证（`held_out_axis_accuracy = 0.0` < 0.9），按预注册 §3.3 **不得写「归因规则可外推到未见方向」**，也不宣称「矫正提升预测准确率」。

## 6. 实测佐证：轴路由随波形族/位置变化

fit 集（周期 128 px）三族 ± 实测轴向角，直接体现映射歧义：

| 族 | 轴 | grating(°) | tilt(°) | blaze_patch(°) |
|---|---|---|---|---|
| x | 面板 x | 1.43 | 0.82 | **-88.73** |
| y | 面板 y | 1.08 | 0.08 | 1.27 |

- **x 族**：grating、tilt 落 CCD **列**轴（≈0°/−180°），blaze_patch 落 CCD **行**轴（≈−88°）——同一面板轴 x，不同波形族路由到不同 CCD 轴。
- **y 族**：grating、tilt、blaze_patch 三族几乎全部落 CCD **列**轴（极差仅 1.18°），面板轴 y 未复现 ≈90° 轴分离。

轴路由随波形族与位置变化，正是 G1/G2/G3 失败的实测根源：#247 的 panel↔CCD 映射**仍歧义、不冻结**。

## 7. 结论

- **归因结论：`not_assessable`**（`anomalous_family = —`）。逐门：G1 fail / G2 fail / G3 fail / G4 fail，未通过的门不被其它门抵消。
- **独立验证**：held-out 门 **fail**，轴向正确 0/10，预测规则 `not_assessable`；未过则不得写任何「归因规则可外推到未见方向」的结论。
- **旧 6 值 verdict `tilt_specific` 仅兼容保存**：本 plan 无 `blaze_*` 图样、周期为 192 px 而非 256 px，其依赖 blaze 与标度比的门会 skipped 或误判，**不得当结论引用**。
- **#247 映射状态**：映射仍歧义、不冻结。本窗口不写正式 `issue247_axis_map.json`，不发布 LUT/效率标定，不声称 #234/#236 有增益。

## 8. 边界

- 本报告只覆盖预注册 v2 窗口的 23 图样、92 帧，结论不外推到其他周期、其他波形族或未测位置。
- 角度口径统一为 `axis_angle_deg`（二值 grating 为 ±1 双峰模 180° 合并后的轴向角，**不是** `principal.angle_deg`）。
- `patch_x_minus_p192` 瓣检测未命中导致无有效角度，是测量缺失而非符号结论。
- 按预注册 §5，本轮不 push、不改 GitHub issue 状态、不写几何标定、不改动 #234/#236 门禁与 transfer `light/none`。
