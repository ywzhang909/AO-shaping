# #247 波形判别窗口预注册（2026-11-23）

> 状态：用户已明确「设备上线，开始采集」，本文件在**下发任何图样之前**写定，作为该窗口的
> 唯一判据来源。窗口结束后不改本文件；与实测不符的部分在结论文档里逐条对照。
>
> 上游计划：`docs/dev/issue247_waveform_diagnostic_plan_20261121.md`
> 上游事实：`docs/dev/agent_notes_20261121.md` §5，`data/debug/hw_window_20261121b/issue247_axis/`，
> `data/debug/hw_window_20261121/issue247_axis_b2d/`

## 1. 窗口身份

| 项 | 值 |
|---|---|
| 设备 | Santec SLM #1（1064 nm）+ Daheng MER2-507-23GM + 1064 nm 激光 |
| 输出目录 | `data/debug/hw_window_20261123_issue247_waveform_probe/`（已存在则失败，拒绝覆盖） |
| 采集脚本 | `scripts/slm/issue247_geometry_capture_20261120.py --plan waveform` |
| 分析脚本 | `scripts/slm/issue247_waveform_probe_solve_20261122.py` |
| 前置门 | `scripts/slm/probe_optical_path_validity.py --session issue247_waveform_20261123`，verdict 必须 `ok` |
| 曝光 / settle | ~~0.05 ms~~ → **0.04 ms** / 0.8 s（采集前修订，见 §8） |
| 帧数 | 4 帧/图样，15 图样 = 60 帧，单次 `--execute`（一次 open/close） |
| 饱和门 | 0.9 × 满量程 |
| 相机 | CCD 全幅 Mono12 |

## 2. 图样顺序（冻结，与计划 §4.1 逐字一致）

| # | ID | 内容 |
|---:|---|---|
| 1 | `flat_pre` | 全平相位（前参考） |
| 2 | `blaze_x_plus` | 连续单轴 ramp，period 128 px，+ |
| 3 | `blaze_x_minus` | 同上，− |
| 4 | `blaze_y_plus` | 同上，y 轴，+ |
| 5 | `blaze_y_minus` | 同上，y 轴，− |
| 6 | `grating_x_p128` | 二值 0/π，p128 |
| 7 | `grating_y_p128` | 二值 0/π，p128 |
| 8 | `tilt_x_plus` | 既有连续 tilt |
| 9 | `tilt_y_plus` | 既有连续 tilt |
| 10 | `flat_mid` | 全平（漂移参考） |
| 11 | `grating_x_p256` | 二值 0/π，p256 |
| 12 | `grating_y_p256` | 二值 0/π，p256 |
| 13 | `blaze_patch_x` | 单个 600×600 patch，period 96（与 ladder 同尺寸/周期） |
| 14 | `blaze_patch_y` | 同上，y 轴 |
| 15 | `flat_post` | 全平（后参考） |

## 3. 已声明的实现事实（写进判据前先承认，避免事后解释）

1. **`blaze_*_±` 与 `tilt_*_±` 是同一个构造函数**（`tilt_phase`，period 128 时既不触发
   `effective_tilt_period` clamp，也不触发 `raw_tilt_feasible` 拒绝；两轴 limit 分别为
   204.8 / 128.0 px，128 均可行）。因此本轮 `blaze_*_±` **不是**「独立构造路径」的对照，
   只能当**同窗口重复性对照**（同一数组隔若干分钟再下一次，用来量热漂移与随机性）。
   计划 §2 中「区分 tilt builder 与 blaze 路径」这一支**在本实现下不可判别**，除非另行
   引入真正的独立构造（不在本轮范围）。
2. 因此本轮真正的新构造只有 **`blaze_patch_x/y`**（ladder 的 patch 机制，只放一个 patch、
   只用一个 order）。它检验计划 H4：patch 位置/局部 ramp 是否让方向脱离 ladder 族结论。
3. 其余图样与 A1b 同源，本轮作用是把 `grating`（二值）与 `tilt/blaze`（连续）在**同一窗口、
   同一 0 级、同一曝光**下再测一次，并给出 drift 参考与 p128/p256 标度。

## 4. 预注册判据（数值与计划 §4.2 相同，此处锁定）

- 预测只用既有冻结映射 A（A1b 的 2×2）作为 ROI 定位输入，**不重新拟合 A**。
- 位移参考 = `flat_pre/flat_mid/flat_post` 的 0 级质心按时间线性插值；三者任一相邻间隔
  质心位移 **> 2 px** → 窗口 `invalid_drift`。
- **波形依赖门**：同轴同周期下，二值 grating 与连续 blaze 的主方向差
  `≤ 2°` → 波形无关；`> 5°` → 波形依赖；`2–5°` → ambiguous。
- **轴区分门**：`blaze_x` 与 `blaze_y` 主方向差须 `88.544° ± 2°`；
  `tilt_x` 与 `tilt_y` 方向差须 `> 80°`，否则 tilt 族判降秩异常。
- **标度门**：`|disp(p128)| / |disp(p256)|` 须 `2.00 ± 0.10`（x、y 各一次）。
- **符号门**：每轴 `+`/`−` 夹角 `180° ± 2°`。
- **饱和门**：任一样本帧饱和像素占比 ≥ 0.9 满量程即记 `saturation`；含饱和的图样其
  位移只作参考，不参与判据。
- 二值 grating 必须**同时**报告 ±1 两侧峰，不允许只挑符合预测的一侧；两侧不等价时
  一并记录 `asymmetric_lobes`。

## 5. 结论枚举（判据算出什么就报什么，不含第五个选项）

`waveform_independent_map_ok` / `binary_grating_specific` / `tilt_specific` /
`patch_position_specific` / `map_not_reproduced` / `invalid_drift_or_saturation`

**无论结果如何**：不写入 `slm_geometry_calib.npz`、不改 `issue247_axis_map*.json`、
不改 #234/#236 双门、不动 transfer `light/none`、不发布 `#238` g₂π。

## 6. 执行纪律

- 段间冷却 ≥ 20 s；六步探测 `verdict=ok` 才允许 `--execute`。
- 一次 open/close；`finally` 顺序 `set_grayscale(0) → cam.close → slm.close → lock.release`。
- 驱动板假死（ReadSU=-1000 等）立即停止，不自动 reopen/retry。
- 结束后确认锁已释放、无残留 bench 进程，并在回复中报告 SLM/CCD/激光占用与释放时间。

## 7. 计划中的偏差记录（本轮实际做了什么、为什么）

- 计划 §3 的「离线准备」在本轮是被实现车道完成而非事前完成：采集脚本扩展与分析器
  与探测并行开发，因此 §3.1 的 phase/gray 哈希核验由脚本 self-test 承担（有限差分
  轴向 + blaze≡tilt 相等断言），阈值全部写死在本文件 §4 而未在看过帧之后调整。
- 计划 §4.1 的 `blaze_x/y_±` 因 §3.1 的事实退化为重复性对照。
- **已执行（2026-11-23）**：六步探测 `20261123_051615Z` verdict=`ok` → 采集 60 帧（0 饱和、
  EXIT=0、设备正常释放）→ 离线分析 verdict=`patch_position_specific`（axis/scale/patch 门未过，
  drift/saturation/waveform 门通过）。实测与判据的逐条对照、原始帧独立核验与结论边界见
  `data/debug/hw_window_20261123_issue247_waveform_probe/WAVEFORM_PROBE_REPORT.md` 与
  `docs/dev/agent_notes_20261123.md` §5；本文件判据未回改。

## 8. 采集前修订：曝光 0.05 → 0.04 ms（2026-11-23，任何图样下发之前）

**触发**：本轮效率匹配窗口中途用户下调了激光功率，A1b 时期验证过的 0.05 ms 不再留有余量。
六步探测（`data/debug/optical_path_probe/issue247_waveform_20261123/20261123_051615Z/`，
`verdict=ok`，噪声底 20.9 counts，饱和线 3890 counts@0.95）实测阶梯：

| 曝光 ms | 平场 0 级峰值 counts | 占本文件 §4 饱和门（3685.5） |
|---:|---:|---:|
| 0.03 | 2366 | 64% |
| 0.04 | 2199 | 60% |
| 0.05 | 3072 | 83% |
| 0.06 | 3053 | 83% |

阶梯非单调（功率慢漂移），且同功率下 20 分钟前的同档读数在 0.05 ms 达 **3688 counts**，
已越过本文件 §4 的 0.9 满量程饱和门。窗口内 flat 帧饱和会毁掉位移参考（§4 首条），
整窗将退化为 `invalid_drift_or_saturation`。

**修订**：曝光改为 **0.04 ms**。理由：两次独立读数下 0.04 ms 峰值 2199 / 2462 counts，
按 20% 漂移上限外推最坏 2954 counts，距 3685.5 门仍有 20% 余量；同时落在探测推荐带
1606–3212 内，质心/FWHM 拟合的 SNR 充足（峰值/噪声 ≈ 105）。settle 0.8 s、帧数 4×15、
饱和门 0.9、全部判据与结论枚举不变——曝光不进入任何几何量的定义（ROI 定位用的冻结映射 A
是几何量，与曝光无关）。

修订时点在**任何图样下发之前**，未使用本窗口任何采集帧选择曝光；采集完成后不回改本节。
