# #247 波形依赖判别实验计划：定位 grating/tilt 与 blaze 轴方向矛盾（2026-11-21）

> 状态：**离线计划，尚未授权上机**。任何 `--execute`、设备占用与 laser/SLM/CCD 开启须等用户新的明确批准。
> 本计划只回答「异常来自波形/灰度链还是面板轴映射」，**不改变**当前「映射仍歧义、不冻结」的结论。
> 依据：`docs/dev/agent_notes_20261121.md` §5，`data/debug/hw_window_20261121b/issue247_axis/`，
> `data/debug/hw_window_20261121/issue247_axis_b2d/`。

## 摘要

同一 session 的两组数据出现系统性矛盾：2-D 对角 blaze（及 ladder）把 panel-x 放在 CCD 行、
panel-y 放在 CCD 列，两轴分离约 88.5°；而 grating/tilt 族把两条面板轴都放到 CCD 列，
`tilt_x` 与 `tilt_y` 仅差 0.46°。这**不是**残差大小问题，而是模型适用域/族间矛盾，
不能靠给正向模型增加矫正参数解决：用 10 个方向再拟合若干补偿系数会把矛盾吸收进参数、
却失去独立验证，按用户的冻结口径仍不通过。

本计划以**波形（灰度分布）为自变量、面板轴为控制变量**做最小判别：同一 panel-x 轴、
匹配空间频率，分别下发二值 0/π grating、连续线性 tilt、局部 blaze patch，
各测 ± 方向，并加 panel-y 对照。若偏转方向随波形而非面板轴变化，指向灰度/LUT/载频链路；
若只有某一族异常，指向该族生成或下发路径。先用未参与拟合的方向独立验证，再谈是否可能
给正向模型加矫正。

## 1. 已知事实（读数据，不重拟合）

角度约定：从 +CCD 列 (dc) 转向 +CCD 行 (dr)；0°≈CCD 列，90°≈CCD 行。

| 探针 | dr, dc (px) | 角度 | 含义 |
|---|---|---|---|
| `grating_x_p128` | −1.90, −95.16 | −178.86° | panel-x 二值 → CCD 列 |
| `tilt_x_plus` | +0.12, +94.68 | +0.07° | panel-x 连续 tilt → CCD 列 |
| `ladder_x` | −88.84, +1.11 | −89.29° | panel-x 局部 blaze → CCD 行 |
| blaze2d (fx 分量) | — | ≈ −90° | panel-x 2-D blaze → CCD 行 |
| `tilt_x_plus` vs `tilt_y_plus` | — | 0.46° | 两 panel 轴在 tilt 族塌成一条 CCD 轴 |

- 二值 grating 只写 `[0, 512]` 两级（0/π）；tilt/ladder/blaze2d 写连续灰度斜坡。
- `grating_x_p128` 与 `tilt_x_plus` 相差约 179°：近似反平行、同一条 CCD 轴；
  与 `tilt_x_minus`（−178.57°）仅差 0.29°。`grating_x_p128` 与 `ladder_x` 相差 89.57°。
- 代码 `axis=1→列` 的约定在三族生成器中一致，但 ladder 与 blaze2d 是独立合成路径。
- flat 0 级偏离几何中心 (+133.8, −67.2 px)；各族使用同一 0 级，差分不影响方向结论。

## 2. 假设与判别表

| 假设 | 预期观测 | 若成立的含义 |
|---|---|---|
| H1 灰度/LUT/γ 非线性 | 二值 grating 方向 ≠ 同轴连续 ramp；周期变化 (p128/p256) 改变二值方向或幅值比 | 查 `#238` 共享 LUT、2π 灰度来源、灰度区间响应；不属于轴映射问题 |
| H2 SLM 固定载频/消零级倾斜 | 所有族整体偏转一个固定 (dr,dc)；panel-x/panel-y tilt 仍近共线 | 查 SLM 驱动叠加相位；需在 0 级参考中扣除载频 |
| H3 tilt/grating 生成或下发路径错轴 | 只有 tilt 或只有 grating 族方向异常；连续单轴 blaze 正常 | 查 phase→gray→面板写入链与 pattern id→axis 绑定 |
| H4 局部 blaze patch 的位置相位被误用 | ladder/blaze patch 方向随 patch 位置变化 | 查 `ladder_phase` 注释所述「patch 位置线性相位于远场不可见」是否成立 |

任何假设成立都**不自动**给正向模型加矫正；先完成第 5 节决策出口。

## 3. 先做的离线准备（零设备）

1. 为下表每个 pattern 保存原始 phase、编码后 gray、shape、axis、频率、相位/灰度 SHA256，
   用有限差分核验 `blaze_x` 只沿列变化、`blaze_y` 只沿行变化，grating/tilt 同理。
2. 用现有 A（仅作预测 ROI 的只读输入）离线生成各探针预测 `(dr,dc)`；统一 0 级局部背景扣除、
   饱和门 0.9、峰定位算法；所有阈值在产生新数据前锁定。
3. 用既有 A1b/b2d 帧离线回放新分析器，确认二值 grating 的 ±1 级简并能被同时报告，
   不允许只挑更符合 A 的一侧。
4. 扩展 capture 入口支持以下冻结图样顺序与 `--patterns` 显式批准范围；未知 id 硬失败。

## 4. 冻结候选设计（待批准后上机）

### 4.1 图样（15 pattern × 4 帧 = 60 帧）

| 顺序 | ID | 内容 | 判别目的 |
|---:|---|---|---|
| 1 | `flat_pre` | 全平相位 | 前参考 |
| 2–5 | `blaze_x_plus/minus`, `blaze_y_plus/minus` | **连续单轴 ramp，period 128 px** | 同一波形直接测两轴基向量与符号 |
| 6–7 | `grating_x_p128`, `grating_y_p128` | 二值 0/π，周期 128 px | 与同轴连续 blaze 对照；测波形依赖 |
| 8–9 | `tilt_x_plus`, `tilt_y_plus` | 既有连续 tilt，effective period | 区分 tilt builder 与 blaze 路径 |
| 10 | `flat_mid` | 全平相位 | 漂移参考 |
| 11–12 | `grating_x_p256`, `grating_y_p256` | 二值 0/π，周期 256 px | 测 1/period 标度与二值选峰口径 |
| 13–14 | `blaze_patch_x`, `blaze_patch_y` | 局部 blaze patch，与 ladder 同尺寸/周期 | 对照 ladder，检验 patch 位置相位 |
| 15 | `flat_post` | 全平相位 | 后参考 |

曝光 `0.05 ms`、settle `0.8 s`、饱和门 `0.9`、4 帧/pattern、CCD 全幅 Mono12，沿用已验证设置。
设备：Santec SLM #1（1064 nm）+ Daheng MER2-507-23GM。一次 open/close 完成窗口，
段间冷却 ≥20 s；若驱动板假死立即停止，不自动 reopen/retry。

### 4.2 预注册判据

- 只用冻结映射 A 预测，**不重新拟合 A**；位移相对 `flat_pre/flat_mid/flat_post` 按时间线性插值。
  平场参考间漂移 >2 px 标记窗口 invalid，不事后挑参考。
- **波形依赖门**：同轴、同频率的二值 grating 与连续 blaze 主方向差 ≤2° 为「波形无关」；
  >5° 为「波形依赖」，触发 H1/H2 排查。
- **轴区分门**：连续 blaze 两轴方向差须 `88.544° ±2°`；tilt 两轴差同样须 >80°，
  否则 tilt 族判为降秩异常（H3）。
- **标度门**：grating p128/p256 偏移量比 `2.00 ±0.10`；blaze/tilt 频率标度同样校验。
- **符号门**：每轴 ± 方向夹角 `180° ±2°`，由未参与“预测 ROI 锁定”的 − 方向作独立验证。
- 结果分类：`waveform_independent_map_ok` / `binary_grating_specific` /
  `tilt_specific` / `patch_position_specific` / `map_not_reproduced` / `invalid_drift_or_saturation`。
  无论结果如何**不自动**改映射、不写 `slm_geometry_calib.npz`。

## 5. 输出与释放

唯一新目录 `data/debug/hw_window_<date>_issue247_waveform_probe/`，拒绝覆盖。
证据至少含 `frames/*.npy`、逐帧 JSONL、`plan.json`、`summary.json`、phase/gray hashes、
`optical_path_record.json`、`released.json`、分析 JSON 与 Markdown。
采集前六步探测 verdict 必须 `ok`；`finally` 顺序 `set_grayscale(0)→cam.close→slm.close→lock.release`，
结束后确认锁释放、无 bench 进程，并在回复中报告 SLM/CCD/激光占用与释放。

## 6. 决策出口（何时才谈正向模型矫正）

1. 连续 blaze 两轴复现 A，仅二值 grating 异常 → 查灰度/LUT/选峰，不动正向模型。
2. tilt 异常但 blaze 正常 → 查 tilt builder / period clamp / 灰度编码，不动正向模型。
3. 波形依赖确认为灰度/LUT/载频 → 先按 `#238` 的共享 LUT 方案分实现分验收；
   只有在**独立窗口**验证后，才评估把已标定响应写入正向模型，且需独立数据证明预测准确率提升。
4. 连续 blaze 也不复现 → 暂停任何写回，先查光路漂移与 pattern→gray→SLM 写入链。
5. 所有门通过 → 作为 #247 映射验证材料提交 owner review；仍不自动冻结。

## 7. 明确不做

- 不用 10 个方向就地拟合补偿系数充当「矫正」：那会把族间矛盾吸收进参数、失去独立验证，
  按用户冻结口径仍不通过。
- 不采 #238 正式 g₂π；不启动 transfer `light/none`；不改闭环 GUI 节拍语义。
- 不声称 #234/#236 有任何增益；不发布新的 #238 g₂π。
- 不改 GitHub issue 状态（评论除外）、不 push，除非另行授权。
