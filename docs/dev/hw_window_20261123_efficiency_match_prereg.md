# 效率匹配设备窗口预注册（2026-11-23）

> 状态：本文件在**下发任何采集帧之前**写定，是该窗口的唯一执行判据来源。窗口结束后不改本文件；
> 与实测不符的部分在结论报告中逐条对照。
>
> 上游计划：`docs/dev/efficiency_match_experiment_plan_20261121.md`
> 判据修订记录：计划文档“2026-11-23 采集前修订回填”节

## 1. 窗口身份与设备占用

| 项 | 值 |
|---|---|
| 设备 | Santec SLM #1（1064 nm）+ Daheng MER2-507-23GM + 1064 nm 激光 |
| 输出目录 | `data/debug/efficiency_match/<UTC>/`（必须为新建 UTC 目录，已存在则采集脚本失败） |
| 采集命令 | `scripts/slm/efficiency_match_capture_20261123.py --execute --out <新目录>`，**单次 execute** |
| 分析命令 | `scripts/slm/efficiency_match_analyze_20261123.py --dir <新目录>` |
| 前置门 | `scripts/slm/probe_optical_path_validity.py --session efficiency_match_20261123`，六步探测 verdict 必须 `ok` |
| 曝光 / settle | **0.04 ms** / 0.8 s，以采集脚本当次读回值为准（采集前修订，见 §4.3；原定 0.05 ms） |
| 段间冷却 | 20 s，实际等待秒数写入 journal |
| 帧数 | 计划 §4 的 8 行 × 2 遍 = 16 帧，单次 `--execute`（一次 SLM/CCD 占用周期） |
| 相机 | CCD 全幅 Mono12，禁止物理开窗 |
| 饱和门 | 0.9 × 12-bit 满量程（4095），记录饱和像素数；饱和帧进入不可判定 |
| 设备锁 | `DeviceLock`；任何驱动板静默回落或假死立即停止，不自动重开 |

## 2. 帧矩阵（冻结）

采集矩阵是计划 §4 表**原样循环两遍**，不是“每种相位各 4 帧”。实际帧数分布：flat 6、
blaze 4、shaping 4、dark 2。第二遍帧 id 带 `_r2` 后缀，journal 用 `raw_file` 区分同 id 的
重复槽位。顺序：

| # | 第一遍 | 第二遍 |
|---:|---|---|
| 1 | `flat_pre` | `flat_pre_r2` |
| 2 | `blaze_grating` | `blaze_grating_r2` |
| 3 | `shaping_target` | `shaping_target_r2` |
| 4 | `dark` | `dark_r2` |
| 5 | `flat_mid` | `flat_mid_r2` |
| 6 | `blaze_grating` | `blaze_grating_r2` |
| 7 | `shaping_target` | `shaping_target_r2` |
| 8 | `flat_post` | `flat_post_r2` |

## 3. 相位与配置冻结

1. **全平相位**：零相位，`uint16=0` 灰度直发，不走 phase-to-gray 的 `mod 2π` 路径。
2. **blaze grating**：单一连续 ramp，`2π/period`，period = **128 面板像元**，axis =
   **panel_x**。不是二值 0/π 光栅。
3. **shaping target**：面板分辨率弧度相位来自
   `data/debug/efficiency_match/shaping_target_panel_20261123.npy`，shape 必须为
   `(1200, 1920)`。SHA256 为
   `f749c3940de902b02d4fa8af6ba708f9834d98741b32fcfed66241a80248116f`；采集时按数组
   `tobytes()` 校验，不符立即失败。该相位由 `data/debug/differentiable_opt/20261117_022658/phase_final.npy`
   经 `phase_model_to_panel()` 最近邻块映射，无菲涅尔补偿（`focus_compensation_focal_mm=inf`）。
4. **暗帧**：原冻结来源 `data/calib/dark/FJB24112232.npz`（2026-11-11，0.05 ms，mean 0.31
   counts）经采集前核查为 Mono8 量级，与本次 Mono12 采集量纲不一致，作废。**采集前修订
   （2026-11-23，见 §4.3）**：按用户决定「复用之前采集的暗帧」，改用相机暗帧库中严格同曝光的
   `data/ccd_configs/ccd/FJB24112232/darks/dark-exp-0.04ms.npz`（12-bit、16 帧、
   mean 14.97 counts），主采集经 `cam.get_dark_frame(0.04, rel_tol=0.02, allow_nearest=False)`
   命中 `dark_source="camera_library"`；不新采暗帧、不跨曝光复用。暗帧来源、复制路径和均值
   必须写入 journal/config；若无法定位同曝光暗帧，其余帧可以落盘，但分析端判 `undeterminable`，
   **不得伪造暗帧**。
5. **ROI 与几何**：采集前已冻结在采集脚本 `ROI_FROZEN`，并在采集时原样写入 `config.json` 的
   `roi_frozen` 块。分析器只读该块，不得自行移动 ROI、掩膜或积分半径。几何来自 2026-11-17
   diffopt 运行实测，`measured_utc=20261117_022701`。

冻结数值摘要：`r0_px=48.0`、`side_px=61.2`、`separation_seed_px=59.1`、
`separation_search_px=25.0`、`r1_px=60.0`、`zero_order_exclusion_px=48.0`、
`saturation_fraction=0.9`、`blaze_period_panel_px=128.0`、`blaze_axis=panel_x`；geometry
含 `kx=7551.141568012036`、`ky=7576.669164768378`、`k_mean=7563.905366390207`、
`rotation_deg=0.16752399978553`、`mirror=false`、`center_cycx=[1104.8427044240864, 1224.4160271278206]`、
`fwhm0=16.0`。周期 128 面板像元的 +1 级预测间距为 59.1 CCD px，这是 `separation_seed_px` 的依据。

## 4. 判据冻结

口径完全沿用计划 §2：

\[
\eta_s=\frac{S-S_0}{F_0}, \qquad \eta_d=\frac{G_1}{F_0},
\]

- \(F_0\) 分母保留完整值，不从分子和分母同时抵消 \(S_0\)；
- \(S\)、\(S_0\)、\(G_1\) 均扣除同一同曝光暗帧贡献；
- 目标掩膜与积分窗口由冻结 ROI 固定；
- 同时报告原始 \(S/F_0\)，但不作最终判定。

匹配判据：`|mean(d)| <= 0.05`，自助 95% CI 上限 `<= 0.08`，且 ηs、ηd 的交错重复组内 sd
分别 `<= 0.03`。

### 4.1 不可判定门

不可判定优先于匹配/不匹配。触发任一项即判 `undeterminable`，不做物理解释，不移动 ROI：

- 漂移轮 > 1 轮（全平参考相对变化超过 5% 的轮次）；
- 任一参与能量积分的帧饱和；
- 全平 0 级不可定位；
- 缺少同曝光暗帧；
- 相位 SHA256 或设备身份与预注册不符；
- 效率越界（不在 0–1 内）；
- **+1 级不可定位**；
- **G1 两轮组内离散超限**。

### 4.2 ±1 对称门修订（采集前修订，不是数据后修）

原计划「±1 级相对差超过 20% → 本次比较作废」**作废**。模型核算显示周期 128 px 的 blaze
光栅在 +1 与 −1 两侧能量天然高度不对称（G1/G−1≈3484:1），20% 对称门必然触发且与物理
构造相矛盾。修订后 ±1 诊断只作为 **blaze 保真度泄漏诊断**：

- 报告 `gm1_over_g1`、`asymmetry_abs_diff_over_max`、`within_expectation`；
- 预期泄漏阈值 `blaze_leakage_expected=0.05`；
- 泄漏超限只标记诊断，**不触发不可判定**；
- 真正作废的 ±1 相关触发是 +1 级不可定位，或 G1 两轮组内离散超限。

修订时点在采集开始前；依据与影响见计划文档回填节。

### 4.3 曝光与暗帧修订（采集前修订，任何采集帧下发之前）

**触发**：2026-11-23 05:00 UTC 起两次六步探测实测，0.05 ms 下平场 0 级峰值已在满量程
（4095/4095，饱和线 3891），原冻结曝光会让 F0 直接截断，ηs/ηd 全部失去意义。

**实测阶梯**（两次探测，均为 12-bit 全幅、噪声底 20.9 counts）：

| 曝光 ms | 降功率前峰值 | 降功率后峰值 |
|---:|---:|---:|
| 0.02 | 2009 | 1409 |
| 0.03 | 3608 | 2404 |
| 0.04 | 3761 | 2462 |
| 0.05 | 4095（饱和） | 3688 |
| 0.06 | — | 3278 |

**修订**：曝光改为 **0.04 ms**。峰值 2462 counts（满量程 60%，距饱和线 37% 余量），
落在目标带 1606–3212 中心附近；0.03 ms 峰值仅高 2.4%，但暗帧库无同档条目。

**暗帧**（按用户 2026-11-23 决定「复用之前采集的暗帧」）：使用相机暗帧库中严格同曝光的
`data/ccd_configs/ccd/FJB24112232/darks/dark-exp-0.04ms.npz`（12-bit，16 帧，
mean 14.97 counts、std 0.895）。`rel_tol=0.02` 下 0.04 ms 为精确命中，不启用 `allow_nearest`；
不需要新采暗帧，也不需要跨曝光复用。暗帧来源与均值必须写入 journal/config；若命中失败，
其余帧照常落盘，分析端判 `undeterminable`，**不得伪造暗帧**。

**影响**：ηs=(S−S0)/F0、ηd=G1/F0 均为同轮同曝光内的比值，曝光从 0.05 改到 0.04 ms 不改变
被测量的定义；判据、帧矩阵、settle、冷却、饱和门一律不变。

## 5. 执行与停止规则

1. 先确认 `data/debug/.device.lock` 无当前持有者，再跑六步光路探测。
2. 探测 verdict 非 `ok`：不进入采集，修复后重跑探测。
3. 只允许一次 `--execute`；新 UTC 目录由采集命令指定，不允许覆盖已有窗口。
4. 采集中发生驱动板假死、锁冲突、曝光读回异常或相位哈希不符：立即进入 `finally` 释放，
   不自动 reopen，本轮采集作废；修复后重新执行完整前置门。
5. `finally` 顺序固定为 `set_grayscale(0) → cam.close() → slm.close() → lock.release()`。
6. 采集结束只检查产物齐备：`raw/`、`phases/`、`journal.jsonl`、`config.json`、
   `release.json`；分析器不得改采集数据。

## 6. 决策出口

- **matched**：满足 §4 全部匹配判据，且无 §4.1 不可判定门触发。
- **mismatched**：无不可判定门触发，但不满足 §4 匹配判据。本结果只描述本固定口径下的比较，
  不能直接归因为 SLM 物理效率不足。
- **undeterminable**：任一 §4.1 不可判定门触发；不写标定、不解禁任何门控、不做物理解释。

## 7. 边界

不写几何标定文件、不判定控制/迁移学习是否解禁、不改 #234/#236 双门、不动 transfer
`light/none`、不发布 #238。本窗口仅回答“固定口径下 ηs 与 ηd 是否匹配”以及不可判定原因。

## 8. 执行结果（2026-11-23，已采集 + 已判读）

- 采集 `data/debug/efficiency_match/20261123_050735Z/`：16 帧全采、0.04 ms、库内同曝光
  暗帧、`saturated_pixels=0`，设备正常释放（§1/§3.4/§4.3 的采前修订均已先行回填）。
- 离线分析 `analysis.json` / `EM_REPORT.md`：verdict = **undeterminable**，触发门
  `energy_out_of_range`（ηs 逐帧为负，即 S < S0，越界部分不作物理解释）、
  `group_sd`（G1 组内离散超限）、`drift_rounds`（4 > 1；F0 跨帧相对变化 4.5–9.4%、
  质心最大位移 3.12 px）。ηd ≈ 0.530–0.553、`|mean d|=0.667`、CI95(d) 高界 −0.639。
- 判据未回改；对照全文见 `docs/dev/agent_notes_20261123.md` §4。
