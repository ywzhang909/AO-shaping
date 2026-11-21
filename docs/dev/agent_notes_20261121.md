# Agent 当日记录 2026-11-21（占用设备：六步预检 + #247 A1 轴映射采集；其余离线）

本轮设备占用：Santec SLM-200 #1 `22030108`（1064 nm，2π 灰度 993）、
Daheng MER2-507-23GM CCD（12-bit 2592×1944）、激光光路。全部已正常释放：
SLM #1 正常 close（16:29:01）、相机经 `finally` 释放、锁 mtime 刷新为释放时刻、
无残留 bench 进程。未 push、未改任何 GitHub issue 状态。

## 1. 六步光路预检（verdict=ok）

`python scripts/slm/probe_optical_path_validity.py --session issue247_20261121 --wait-until-idle 900`
→ exit 0，verdict `ok`。证据 `data/debug/optical_path_probe/issue247_20261121/20261121_081053Z/`。
判读要点：可用曝光 0.05 ms（峰值 1834 counts、噪底 20.9）；离焦响应半宽 +4.08 px、+1 rad 峰值 +477.8 counts；SLM 真在调制。

## 2. #247 A1 轴映射采集（已完成，结论=映射仍歧义，未冻结）

- 命令要点：`scripts/slm/issue247_geometry_capture_20261120.py --out data/debug/hw_window_20261121/issue247_axis`
  `--exposure-ms 0.05 --settle-s 0.8 --saturation-fraction 0.9`（4 帧均值）。
- 结果：`status=ok frames=44`（11 图案 × 4 帧），曝光 0.05 ms 平地峰值 ≈1717–1923 counts（≈42–47%，未饱和）。
- 0 级真实位置 (row 1105.25, col 1227.56)，相对几何中心偏 (dr +133.25, dc −68.44)；几何中心 (972,1296) 是背景；
  需排除 (r≈8,c≈8) 角落杂散。

**分析**：`scripts/slm/issue247_axis_map_solve_20261121.py` → `axis_mapping_status.json`
（同目录 `README_axis_mapping.md` 为可读版）。两法（局部极大 + 平地扣除能量投影）互证，
探针分裂成两组、彼此差 ~90°：

- **阶梯 family 与历史 88.5° 旋转自洽**：`ladder_x`(panel-x)→CCD **行**，`ladder_y`(panel-y)→CCD **列**。
- **光栅 + 倾斜 family 不自洽**：`grating_x/grating_y`、`tilt_x/tilt_y` 不论 panel 轴都落到 CCD **列**
  （`tilt_x` vs `tilt_y` 仅差 0.63°，`grating_x` vs `grating_y` 同理）。两个正交 panel 轴映到同一条 CCD 线，
  任何可逆 2×2 映射都做不到。

判据（同轴探针方向差 ≤2°、两 panel 轴方向差 ≈88.544°）**不满足** →
`frozen=false, verdict=mapping_still_ambiguous`，**未写 `issue247_axis_map.json`**，#247 保持 open。
逻辑/显示侧已排除：三 builder 均 `axis=1→列/axis=0→行`，显示链 `pattern_array→phase_to_gray→slm.display_data` 无转置。

下一步（需新的上机批准，不在本轮）：采 2-D 对角 blaze（`blaze2d_sx+1_sy+1` 与 `sx+1_sy-1`，周期 128/192 px，
A0-3 已实现），单个离轴对角点一次读出两轴、无对称对歧义；再复采一拍 `grating_x_p128` 确认 ±列结果稳定。
残差 RMS ≤1.0 px、相对 ≤2% 才可冻结。

## 3. A0-3 离线合成验收自检（已修复，accepted=True）

`scripts/slm/issue247_synthesis_acceptance_20261121.py`。剩余失败项 `blaze2d_sx+1_sy-1 predicted spots not confirmed by FFT`
根因：blaze2d 是整面连续 ramp（单位幅度），能量全进单一 1 级，DC 恰为数学零点
（`1920/128=15` 为整数，`exp(i·ramp)` 整面和为 ~1e-23），旧代码却让 `verify_spots` 去找不存在的 (0,0) 局部极大。
修法：`predict_spots` 的 blaze2d 分支不再预测 (0,0)（阶梯用局域 patch 有真实 DC，保留）。
- `--self-test` exit 0：`all invariants passed (13 candidates, accepted=True)`。
- `--out data/debug/slm_transfer/issue247_synthesis_20261121.json`：`accepted=True all_fft_confirmed=True`，
  两个 blaze2d 均 `fft_confirmed:true`。`VERIFY_RADIUS_BINS` 仍为 2、`verify_spots` 未改，无放宽。

## 4. 仍阻塞项（沿用 2026-11-20 结论，本轮未动）

- **B1 / #238 正式 g₂π**：`dark_capture_hook` 无硬件熄灯实现，`--dark-frames N>0` 直接 fail-fast；
  需先接照明熄灯 + 再次上机批准。未发布任何新 g₂π。
- **#234 / #236 双门复跑**：离线仍失败，修复门禁后再谈上机；未据既有数据宣称增益。

## 5. #247 A1b 复采 + 2-D 对角 blaze（verdict=mapping_frozen）

本轮设备占用：Santec SLM #1（1064 nm）+ Daheng CCD + 激光。六步探测 verdict=`ok`（17:37），
20 s 冷却后单次 `--execute` 采集（17:41–17:44），`status=ok frames=60`（15 图样 × 4 帧），
`set_grayscale(0)→cam.close→slm.close→lock.release`，锁 mtime 刷新、无残留 bench 进程。

- 预注册：`docs/dev/hw_window_20261121b_issue247_prereg.md`（采集前冻结；段间冷却按用户要求
  由 200 s 改为 20 s，其余口径/判据不变）。目录 `data/debug/hw_window_20261121b/issue247_axis/`；
  未改动上午 `hw_window_20261121/`，未写 `slm_geometry_calib.npz`。
- 离线代码：`issue247_geometry_capture_20261120.py` 增加 blaze2d 图样（唯一 `blaze2d_phase`
  实现，synthesis 脚本改为委托），15 图样/60 帧，`--self-test` exit 0；
  `issue247_axis_map_solve_20261121.py` 增加 2×2 最小二乘与冻结判定，旧目录复跑仍
  `mapping_still_ambiguous` 且记录产物未改；ruff 全过。
- **解算**（`axis_mapping_status.json` / `README_axis_mapping.md` / `issue247_axis_map.json`）：
  `frozen=true, verdict=mapping_frozen`。四项判据全 PASS——残差 RMS **0.871 px**（≤1.0）、
  最大相对误差 **1.576 %**（≤2 %）、双列夹角 **90.36°**（88.544±2°）、秩守卫
  **0.9925**（≥0.10）；拟合 A≈[[+26.3, −7673.4],[+7705.9, −22.0]]（CCD px per cycle/panel-pixel），
  cond≈1.01。
- **归因**：blaze2d（四方向独立点）支持 ladder family——panel-x ramp→CCD 行、panel-y ramp→CCD 列；
  grating/tilt family 的"两正交轴都落 CCD 列"与任何可逆 2×2 映射不相容。当前仅能定位为
  grating/tilt 探针输出与设计映射不一致，具体根因待独立验证；辅助拟合：grating fy/fx 列向
  −178.7°/−88.4°，tilt +1.0°/+90.9°，blaze2d 为参考。这与历史 88.544° 旋转自洽，但
  `focal_magnification` 仍为反推，不宣称"物理常数已定"。
- 未改 #247 issue 状态、未 push；仅按冻结方案 §3.4 发一条事实性评论。未动 #234/#236 双门、
  未动 transfer `light/none` 暂停判定。
