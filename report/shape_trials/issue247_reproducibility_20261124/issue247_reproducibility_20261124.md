# Issue #247 同窗口重复性复核（离线检查）

<!-- provenance:start -->
> **生成脚本**: [`scripts/slm/issue247_reproducibility_check_20261124.py`](../../../scripts/slm/issue247_reproducibility_check_20261124.py)
> **复现命令**: `python scripts/slm/issue247_reproducibility_check_20261124.py --dir data/debug/hw_window_20261124_issue247_reproducibility_v2 --baseline-dir data/debug/hw_window_20261123_issue247_waveform_attribution --probe-dir data/debug/optical_path_probe/issue247_reproducibility_20261124_v2/20261124_103335Z --out-dir report/shape_trials/issue247_reproducibility_20261124`
> **数据/关联脚本**: [`scripts/slm/issue247_geometry_capture_20261120.py`](../../../scripts/slm/issue247_geometry_capture_20261120.py)
> **数据/关联脚本**: [`scripts/slm/probe_optical_path_validity.py`](../../../scripts/slm/probe_optical_path_validity.py)
> **运行环境**: 离线+硬件（Santec SLM-200 #1 1064 nm + Daheng MER2-507-23GM cam 0，2026-11-24 采集；报告本体不占用设备）
> **说明**: Issue #247 波形归因信号同窗口重复性复核（预注册 hw_window_20261124_issue247_reproducibility_prereg.md）：同一 SLM 打开周期内 23 图样 × 4 帧 × 3 轮（frozen / reversed / frozen）= 69 唯一图样 / 276 帧，曝光 0.04 ms、settle 0.8 s、饱和门 0.9 × full_scale、零饱和帧；每轮独立重跑同一分析器。方向重复性门（至少两轮可评估且任两轮 axis_angle_deg 差 ≤ 2°）：复现 15 / 未复现 0 / 缺测 5，overall_pass=True；可评估项轮间最大轴角差 1.343°（tilt_y_plus），flat 跨轮最大漂移 0.469 px。5 个 p192 图样因 principal 峰心落在帧边界 8 px 带内（peak 21.75–25.0 counts / 背景 15.5）按预注册保留为缺测、不调阈值救回；逐轮 G1–G4 全 fail（verdict=tilt_specific），与 v2 归因窗口一致。本轮只回答同窗口重复性，不冻结映射、不证明根因、不关闭 issue。
<!-- provenance:end -->

- **数据**: `data/debug/hw_window_20261124_issue247_reproducibility_v2`
- **基线**: `data/debug/hw_window_20261123_issue247_waveform_attribution`
- **预检 run**: `data/debug/optical_path_probe/issue247_reproducibility_20261124_v2/20261124_103335Z`
- **分析器**: `scripts/slm/issue247_waveform_probe_solve_20261122.py`
- **预注册**: `docs/dev/hw_window_20261124_issue247_reproducibility_prereg.md`
- **生成时间**: 2026-11-24T11:32:19.160783+00:00

## 结论先行

- 输入冻结项：全部通过（3 轮 × 23 图样 × 4 帧 = 276 帧）。
- 方向重复性门（任意两轮轴角差 ≤ 2.0°，只用 `axis_angle_deg`）：**pass**；reproduced=15，not_reproduced=0，not_assessable=5（其中边界极大值排除 15 轮）；可评估轮次合计 45。
- 逐轮 G1–G4：r1: G1=fail, G2=fail, G3=fail, G4=fail；r2: G1=fail, G2=fail, G3=fail, G4=fail；r3: G1=fail, G2=fail, G3=fail, G4=fail。
- 边界：本报告只说明同一硬件打开周期内是否重复；不证明根因、不冻结映射、不代表跨会话独立复现。

## 1. 输入与冻结项校验

| 项 | 值 |
|---|---|
| 轮次 | r1, r2, r3 |
| 每轮图样数 | 23 |
| 唯一 pattern ID | 69 |
| 实际帧数 | 276 |
| 曝光 | 0.040 ms |
| settle | 0.800 s |
| 每图样帧数 | 4 |
| full scale | 4095.0 |
| 帧 shape | 1944 × 2592 px |
| 饱和门 | 3685.5 counts |
| flat 跨轮最大漂移 | 0.469 px (≤ 2.0) |
| 冻结顺序来源 | frozen_issue247_attribution_order |
| 相位摘要 | 23 个 base pattern 三轮一致，基线核对 23 个 |
| 设备释放 | status=ok，closed_normally=pass |
| 采集光路记录 | 已复核 11 项 |
| 六步预检 | verdict.state=ok，exit_code=0 |
| 预检可用曝光窗口 | 0.040, 0.050, 0.060 ms（冻结 0.040 ms 在内） |

**证据来源**

- `capture_plan`：plan.json：轮次/顺序/相位摘要/帧数/capture_settings.settle_s/exposure_ms
- `capture_summary`：summary.json：状态/总帧数/饱和计数/逐图样帧数
- `capture_released`：released.json：SLM/相机/锁正常释放
- `capture_optical_record`：optical_path_record.json：设备身份与exposure_check（六步预检 verdict 不在其中）
- `probe_run`：optical_path_probe.json：六步预检 verdict.state、exit_code、可用曝光窗口
- `baseline`：waveform_probe_analysis.json / plan.json / summary.json / released.json：attribution 基线

## 2. 每轮 G1–G4（逐轮单独计算）

| 门槛 | r1 | r2 | r3 |
|---|---|---|---|
| G1_waveform_dependency | fail (x: 89.83°; y: 0.78°) | fail (x: 89.60°; y: 0.73°) | fail (x: 89.94°; y: 1.61°) |
| G2_single_family_anomaly | fail (anomalous=none) | fail (anomalous=none) | fail (anomalous=none) |
| G3_axis_separation | fail (grating: 0.13°; tilt: 0.05°; blaze_patch: 89.83°; routing_failed=grating,tilt) | fail (grating: 0.63°; tilt: 0.07°; blaze_patch: 89.60°; routing_failed=grating,tilt) | fail (grating: 0.65°; tilt: 0.13°; blaze_patch: 89.10°; routing_failed=grating,tilt) |
| G4_sign_consistency | fail | fail | fail |

| 轮次 | verdict | 失败 gates | held-out 轴向准确率 |
|---|---|---|---|
| r1 | tilt_specific | axis_gate, scale_gate, sign_gate | — |
| r2 | tilt_specific | axis_gate, scale_gate, sign_gate | — |
| r3 | tilt_specific | axis_gate, scale_gate, sign_gate | — |

## 3. 逐 pattern 方向重复性（axis_angle_deg，模 180°）

| base pattern | 族 | 角色 | r1 | r2 | r3 | 帧内极差 r1/r2/r3 | 轮间最大差 | 状态 | 缺测 |
|---|---|---|---|---|---|---|---|---|---|
| grating_x_p128 | grating | fit | 1.205 | 0.994 | 1.385 | 1.18 / 0.35 / 0.98 | 0.39° | reproduced | — |
| tilt_x_plus | tilt | fit | 1.048 | 1.034 | 1.092 | 1.07 / 0.87 / 0.69 | 0.06° | reproduced | — |
| tilt_x_minus | tilt | fit | -179.408 | -179.024 | -178.758 | 0.45 / 0.66 / 0.85 | 0.65° | reproduced | — |
| patch_x_plus_p128 | blaze_patch | fit | -87.475 | -88.357 | -88.342 | 1.41 / 0.09 / 2.12 | 0.88° | reproduced | — |
| patch_x_minus_p128 | blaze_patch | fit | 90.226 | 91.157 | 91.237 | 3.43 / 1.70 / 0.91 | 1.01° | reproduced | — |
| grating_y_p128 | grating | fit | 1.075 | 1.620 | 0.738 | 0.57 / 1.20 / 1.14 | 0.88° | reproduced | — |
| tilt_y_plus | tilt | fit | 0.853 | 0.766 | 2.109 | 1.29 / 0.34 / 2.52 | 1.34° | reproduced | — |
| tilt_y_minus | tilt | fit | -179.316 | -178.614 | -179.513 | 2.63 / 1.95 / 1.16 | 0.90° | reproduced | — |
| patch_y_plus_p128 | blaze_patch | fit | -6.994 | -7.081 | -5.953 | 1.29 / 2.70 / 2.26 | 1.13° | reproduced | — |
| patch_y_minus_p128 | blaze_patch | fit | -169.907 | -169.314 | -169.351 | 3.12 / 5.70 / 1.37 | 0.59° | reproduced | — |
| grating_x_p192 | grating | held_out | 0.725 | 0.979 | 1.540 | 1.32 / 1.04 / 0.61 | 0.82° | reproduced | — |
| tilt_x_plus_p192 | tilt | held_out | 1.388 | 0.171 | 0.453 | 2.28 / 1.28 / 1.55 | 1.22° | reproduced | — |
| tilt_x_minus_p192 | tilt | held_out | -178.309 | -178.195 | -178.581 | 2.38 / 1.10 / 2.27 | 0.39° | reproduced | — |
| patch_x_plus_p192 | blaze_patch | held_out | -86.655 | -87.703 | -87.576 | 1.98 / 2.79 / 3.85 | 1.05° | reproduced | — |
| patch_x_minus_p192 | blaze_patch | held_out | -124.838† | -124.853† | -124.855† | 0.04 / 0.02 / 0.02 | —° | not_assessable | r1:edge_clipped, r2:edge_clipped, r3:edge_clipped |
| grating_y_p192 | grating | held_out | 1.223 | 1.349 | 1.024 | 0.54 / 0.38 / 0.31 | 0.33° | reproduced | — |
| tilt_y_plus_p192 | tilt | held_out | -124.858† | -124.861† | -124.848† | 0.03 / 0.04 / 0.08 | —° | not_assessable | r1:edge_clipped, r2:edge_clipped, r3:edge_clipped |
| tilt_y_minus_p192 | tilt | held_out | -124.842† | -124.847† | -124.839† | 0.04 / 0.06 / 0.02 | —° | not_assessable | r1:edge_clipped, r2:edge_clipped, r3:edge_clipped |
| patch_y_plus_p192 | blaze_patch | held_out | -124.829† | -124.844† | -124.871† | 0.04 / 0.03 / 18.96 | —° | not_assessable | r1:edge_clipped, r2:edge_clipped, r3:edge_clipped |
| patch_y_minus_p192 | blaze_patch | held_out | -124.836† | -124.861† | -124.854† | 0.03 / 0.05 / 0.05 | —° | not_assessable | r1:edge_clipped, r2:edge_clipped, r3:edge_clipped |

缺峰保留为缺测，不调阈值；完整逐帧质心、候选峰、峰数与峰/背景比见 JSON 的 `rounds[].analysis.patterns` 与 `repeatability.patterns[].per_round[]`。

### 3.1 判读边界条件（重复性门的排除项）

- 规则：principal 峰心落在帧任一边界 8 px 带内，即视为被边界截断的极大值（不是波瓣）；该轮记为缺测，不进入 ≤2.0° 重复性门。原始 axis_angle_deg、principal、peak、background、峰心仍完整保留在 JSON。
- 本次帧 shape：1944 × 2592 px；被排除的图样 5 个、轮次 15 个：
  - `patch_x_minus_p192` r1：principal peak=24.5 counts / background=15.5 counts，峰心 row=1.4, col=460.3（距最近边界 1.4 px），原始 axis_angle_deg=-124.84° 已排除。
  - `patch_x_minus_p192` r2：principal peak=23.0 counts / background=15.5 counts，峰心 row=1.6, col=460.1（距最近边界 1.6 px），原始 axis_angle_deg=-124.85° 已排除。
  - `patch_x_minus_p192` r3：principal peak=22.5 counts / background=15.5 counts，峰心 row=1.4, col=460.0（距最近边界 1.4 px），原始 axis_angle_deg=-124.86° 已排除。
  - `tilt_y_plus_p192` r1：principal peak=24.2 counts / background=15.5 counts，峰心 row=1.7, col=459.9（距最近边界 1.7 px），原始 axis_angle_deg=-124.86° 已排除。
  - `tilt_y_plus_p192` r2：principal peak=22.8 counts / background=15.5 counts，峰心 row=1.4, col=459.7（距最近边界 1.4 px），原始 axis_angle_deg=-124.86° 已排除。
  - `tilt_y_plus_p192` r3：principal peak=21.8 counts / background=15.5 counts，峰心 row=1.5, col=460.4（距最近边界 1.5 px），原始 axis_angle_deg=-124.85° 已排除。
  - `tilt_y_minus_p192` r1：principal peak=22.0 counts / background=15.5 counts，峰心 row=1.7, col=460.3（距最近边界 1.7 px），原始 axis_angle_deg=-124.84° 已排除。
  - `tilt_y_minus_p192` r2：principal peak=24.0 counts / background=15.5 counts，峰心 row=1.2, col=460.0（距最近边界 1.2 px），原始 axis_angle_deg=-124.85° 已排除。
  - `tilt_y_minus_p192` r3：principal peak=25.0 counts / background=15.5 counts，峰心 row=1.4, col=460.5（距最近边界 1.4 px），原始 axis_angle_deg=-124.84° 已排除。
  - `patch_y_plus_p192` r1：principal peak=23.8 counts / background=15.5 counts，峰心 row=1.5, col=460.6（距最近边界 1.5 px），原始 axis_angle_deg=-124.83° 已排除。
  - `patch_y_plus_p192` r2：principal peak=24.8 counts / background=15.5 counts，峰心 row=1.3, col=460.2（距最近边界 1.3 px），原始 axis_angle_deg=-124.84° 已排除。
  - `patch_y_plus_p192` r3：principal peak=23.5 counts / background=15.5 counts，峰心 row=1.7, col=459.8（距最近边界 1.7 px），原始 axis_angle_deg=-124.87° 已排除。
  - `patch_y_minus_p192` r1：principal peak=23.0 counts / background=15.5 counts，峰心 row=1.6, col=460.5（距最近边界 1.6 px），原始 axis_angle_deg=-124.84° 已排除。
  - `patch_y_minus_p192` r2：principal peak=21.8 counts / background=15.5 counts，峰心 row=1.4, col=459.8（距最近边界 1.4 px），原始 axis_angle_deg=-124.86° 已排除。
  - `patch_y_minus_p192` r3：principal peak=22.2 counts / background=15.5 counts，峰心 row=1.3, col=460.0（距最近边界 1.3 px），原始 axis_angle_deg=-124.85° 已排除。
- 基线对照（基线 principal 峰位/峰值）：
  - `patch_x_minus_p192`：基线 principal 为空（基线口径下该图样即缺测）。
  - `tilt_y_plus_p192`：基线 principal magnitude=40.2 px、peak=795.5 counts，即真实波瓣位于 0 级约 40 px 处；本次逐轮记录 core_exclude_px=41.0 px，大于该距离，因此本次口径下该波瓣被 0 级核心排除。
  - `tilt_y_minus_p192`：基线 principal magnitude=39.5 px、peak=966.8 counts，即真实波瓣位于 0 级约 40 px 处；本次逐轮记录 core_exclude_px=41.0 px，大于该距离，因此本次口径下该波瓣被 0 级核心排除。
  - `patch_y_plus_p192`：基线 principal magnitude=39.3 px、peak=188.5 counts，即真实波瓣位于 0 级约 39 px 处；本次逐轮记录 core_exclude_px=41.0 px，大于该距离，因此本次口径下该波瓣被 0 级核心排除。
  - `patch_y_minus_p192`：基线 principal magnitude=37.2 px、peak=258.5 counts，即真实波瓣位于 0 级约 37 px 处；本次逐轮记录 core_exclude_px=41.0 px，大于该距离，因此本次口径下该波瓣被 0 级核心排除。
- G1–G4 的输入里这 5 个图样的轴角是边界极大值伪值；三轮 G1–G4 仍全 fail，结论不依赖这些伪值。`rounds[*].analysis.gates/verdict` 仍是分析器原值，本节排除只作用于重复性门。

## 4. 与归因基线对照（仅报告，不作重复性门）

- 基线 verdict：`tilt_specific`
- 基线相位核对：23 个 base pattern
- 逐 pattern 的基线与各轮轴角差见 `repeatability.patterns[].baseline`；基线不同会话，不参与 ≤2° 重复性判定。

## 5. 限制与结论边界

- 本检查只判断同一硬件打开周期内的重复性；不证明根因、不冻结 panel→CCD 映射、不代表跨会话独立复现。
- 重复性门通过只能说该异常在本次连续硬件会话内可重复；不等同于跨会话独立复现。
- 若未通过，结论为本窗口信号不稳定/证据不足，不等同于 issue 无效。
- panel-y 轴路由、实际 LUT/相位写回、匹配支持域及未见方向验证仍为未解决证据缺口。
