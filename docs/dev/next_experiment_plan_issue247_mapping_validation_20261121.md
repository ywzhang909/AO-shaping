# 下一步实验计划：#247 轴映射独立验证与 grating/tilt 异常定位（2026-11-21）

> 状态：**离线计划草案，尚未授权执行**。任何 `--execute` 或设备占用须等用户新的明确批准。
> 依据：`docs/dev/hw_window_20261121b_issue247_prereg.md` 与
> `data/debug/hw_window_20261121b/issue247_axis/` 的 A1b 数据。

## 摘要

本轮 2-D blaze 拟合通过预注册判据（RMS 0.871 px、最大相对误差 1.576%、轴夹角 90.36°、秩守卫 0.9925），
得到一个数值稳定、与历史旋转自洽的描述性 panel→CCD 映射。但它仍来自**同一设备窗口、同一批图样**，
还不是独立复现，也不应直接写入 `slm_geometry_calib.npz`。

下一步不重测整套 A1b，而是用**独立的 1-D blazed ramps**直接测出映射的两列，并与旧二值 grating / tilt
探针在同一窗口交错对照。目标是回答：单轴连续 blaze 与 2-D blaze 是否复现相同基向量？若复现，旧 grating/tilt
异常来自二值图样的衍射/选峰口径，还是 tilt builder / 显示链路？

## 1. 已知事实与边界

- A1b 的四个 2-D blaze 单点分别位于四个非共线频率；最小二乘映射通过四项冻结判据。
- `ladder_x` / `ladder_y` 的方向与 2-D blaze 映射一致；上午与本轮的 grating/tilt 方向读数仍显示两 panel 轴近似共线。
- A1b `issue247_axis_map.json` 是本轮描述性映射；`focal_magnification` 仍为反推量。
- 下一实验**只验证与归因**：不重估物理标定常数、不写 `slm_geometry_calib.npz`，不重判 #234/#236 双门，
  不改变 transfer `light/none` 暂停状态，也不作为任何控制/模型解禁证据。

## 2. 先做的离线准备（零设备）

1. 对每个图样生成并保存原始 phase、编码后的 gray、shape、axis、频率、相位/灰度 SHA256；用有限差分核验
   `blaze1d_x` 只沿列变化、`blaze1d_y` 只沿行变化，tilt / grating 同理。
2. 将本轮拟合矩阵 `A` 作为只读输入，离线生成各探针的预测 `(dr,dc)` 与成像搜索 ROI；所有图样使用统一的
   0 级局部背景扣除、饱和门与峰定位，不允许看过新数据后改 ROI/阈值。
3. 编写新的、纯描述性 capture/analyze 入口或扩展现有入口：支持冻结的图样顺序、重复 `flat` 参考、
   明确写出相位哈希与每帧元数据；保留 `--execute` 显式硬件门、唯一新输出目录、DeviceLock、释放记录。
4. 用合成图与 A1b 既有帧完成离线自测；先冻结下表、算法版本与判据，再申请设备窗口。

## 3. 下一次窗口的冻结候选设计

### 3.1 图样（15 个 pattern × 4 帧 = 60 帧）

顺序交错，既有 flat 参考用于观测漂移，不允许以新数据调整顺序：

| 顺序 | ID（建议） | 内容 | 诊断目的 |
|---:|---|---|---|
| 1 | `flat_pre` | 全平相位 | 前参考 |
| 2–5 | `blaze1d_x_plus/minus`, `blaze1d_y_plus/minus` | 1-D 连续 ramp，period 128 px，各轴正负号 | 直接测 A 的两列与符号反向；避免二值光栅多级衍射 |
| 6–7 | `grating_x_p128`, `grating_y_p128` | 二值 0/π，周期 128 px | 与 1-D 连续 blaze 同轴对照 |
| 8–9 | `tilt_x_plus`, `tilt_y_plus` | 既有单轴线性 tilt，沿用 A1b effective period | 检验 tilt builder 是否跟 1-D blaze 一致 |
| 10 | `flat_mid` | 全平相位 | 中途漂移参考 |
| 11–12 | `grating_x_p256`, `grating_y_p256` | 二值 0/π，周期 256 px | 检查 1/period 标度与选峰口径 |
| 13–14 | `tilt_x_minus`, `tilt_y_minus` | 与 plus 配对 | 检查同轴正负是否反向且幅值相当 |
| 15 | `flat_post` | 全平相位 | 后参考 |

曝光 `0.05 ms`、settle `0.8 s`、饱和门 `0.9`、4 帧/pattern、CCD 全幅 Mono12，沿用本轮已验证设置。
设备 SLM #1（1064 nm）+ Daheng MER2-507-23GM；仍按一次 open/close 完成窗口。

### 3.2 预注册分析与判据

- 只用冻结映射 A 预测，不重新拟合新 A；所有位移相对 `flat_pre/flat_mid/flat_post` 按时间线性插值，
  每帧单独保存位置与误差。平场参考间漂移若超过 2 px，标记窗口 invalid，不事后挑参考。
- **1-D blaze 基向量复现门**：每个轴的正负号位移须沿 A 对应列方向（方向误差 ≤2°）；正负位移夹角
  `180° ±2°`；每个轴四帧均值的预测残差 ≤1.0 px 且相对误差 ≤2%。两轴夹角须为 `88.544° ±2°`。
- **tilt 对照门**：每个 tilt 位移方向与同轴 1-D blaze 相差 ≤2°，正负方向相反，幅值差相对 ≤2%。
- **grating 标度/方向诊断**：每轴 p128 与 p256 的主峰方向对照该轴 blaze，方向差 ≤2°；偏移量比应为 `2.00 ±0.10`。
  由于 binary grating 正负级简并，须同时报告 +1/−1 两侧的峰位置/强度，不允许只挑更符合 A 的一侧。
- 整体结果只分为：`mapping_reproduced_and_legacy_families_consistent`、
  `mapping_reproduced_but_binary_grating_specific_anomaly`、`mapping_reproduced_but_tilt_specific_anomaly`、
  `mapping_not_reproduced`、`invalid_drift_or_saturation`。无论结果如何都不自动更改映射或设备标定。

> 这些阈值在离线自测后、任何新硬件数据产生前最终锁定。若离线估计表明 1 px 高于测量噪声无法达到，
> 必须在预注册更新中用已有数据给出推导并由用户批准；不得看新数据后放宽。

## 4. 硬件窗口、输出与释放

- 预计约 3 min 采集；采集前重新执行六步探测，verdict 必须 `ok`，曝光/光通量异常则中止。
- 输出：`data/debug/hw_window_<date>_issue247_validation/`，唯一新目录，拒绝覆盖。
- 证据至少包含 `frames/*.npy`、逐帧 JSONL、`plan.json`、`summary.json`、phase/gray hashes、
  `optical_path_record.json`、`released.json`、分析 JSON 与 Markdown。
- 段间冷却按用户本次要求为 ≥20 s；鉴于曾记录 SLM 背靠背 open/close 假死风险，若出现驱动板异常，
  立即停止，不自动 reopen/retry。
- `finally` 顺序 `set_grayscale(0)→cam.close→slm.close→lock.release`；结束后确认锁释放、无 bench 进程。

## 5. 决策出口

1. 1-D blaze 两轴都复现 A，但 grating 仍异常：优先离线查二值 grating 频谱/±级选峰实现；不再用 grating 结果拟合物理映射。
2. blaze 复现、tilt 异常：审查 `tilt_geometry/tilt_phase` 与 period clamp / 灰度编码，不实验扫新参数。
3. 1-D blaze 不复现：暂停任何标定写回；检查光路时间漂移、pattern→gray→SLM 写入数据链，先离线复查再提新的受控验证。
4. 所有门通过：把本轮与独立窗口的证据作为 #247 映射验证材料；仍需 owner review 后，另行决定是否将映射转成设备校准记录。

## 6. 明确不做

- 不采 #238 正式 g₂π（暗场熄灯 hook 尚未接线）；不启动 transfer `light/none`。
- 不把节拍解耦或本实验描述成实验提速；不改闭环 GUI 串行语义。
- 不改 GitHub issue 状态、不 push、不生成 `slm_geometry_calib.npz`，除非另行授权。
