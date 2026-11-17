# 实验方向门槛测试计划（2026-11-17）

> **📌 状态以合并版为准**：本文是 P1–P5 方向门槛的原子计划，已被
> [`hw_plan_consolidated_20261118.md`](hw_plan_consolidated_20261118.md) 汇编为 2f 整形唯一入口。
> 完成标记、设备段取舍与执行顺序以合并版 §1–§4 为准；本文保留作判据与命令细节依据。
> 关键变化：P1/P2a 采集+分析已完成、P2b 数据已落盘只差离线判读，P3/P4 受决策门 D1 门控（可能整段推迟）。

> 状态：待执行（激光暂停中，先备好计划与代码）。一次实验、**一次** SLM/CCD open/close
> 走完全部阶段，避免背对背 open/close 触发驱动板假死（`ReadSU=-1000`）。
> 纪律见 `docs/dev/hardware_discipline.md`；六步光路探测见
> `libs/skills/ao-optical-path-calibration/SKILL.md`。

## 0. 设备与预算

| 项 | 值 |
|---|---|
| SLM | Santec SLM-200 #1，1064 nm，驱动板 `22030108` |
| CCD | Daheng MER2-507-23GM，cam_id 0 |
| 位深 | **Mono12**（8-bit 下 1 count ≈ 7%，不能用于均匀度结论） |
| 曝光 | 0.05 ms（差分刀口）/ 0.10 ms（探针帧）；任一帧峰值 ≥ 0.98 满量程即降曝光重来 |
| 开窗 | **不做物理开窗**，全幅采集，只在后处理裁切 |
| 锁 | 全程 `DeviceLock`；启动加 `--wait-until-idle 600` |
| 释放 | 每个 `--execute` 脚本自带 finally：cam.close → slm.close → lock.release → `released.json` |

前置检查（失败即停，不进入下一阶段）：

1. `DeviceLock.default().holder()` 返回 None；
2. `Santec(1, video_mode=0).open()` 成功且 `require_drive_board_entry(slm)` 通过，
   `slm.serial_source() == "driveboard"`；
3. 相机 open 后 `effective_full_scale(img)` 与 `bit_depth` 一致；
4. 零相位帧有 PSF 焦点（A=0 聚焦），不是弥散圆斑；
5. 相位探针自检通过（`probe_status` 必须在驱动板已应答的前提下才可信）。

## 本轮执行范围

用户已定：**先跑 P1 + P2**，P3 / P4 待 P2 结论出来后再定。

三个 `--execute` 脚本各自 open/close 一次 SLM+CCD，**背对背 open/close 会触发驱动板
假死**（`ReadSU=-1000`），因此三段之间必须插入冷却（≥ 3 min），并在每段开始前重新核对
`require_drive_board_entry(slm)` 与 `slm.serial_source() == "driveboard"`。

## 1. 阶段 P1 — 零级定心（#235）

目标：拿到同一批帧上四种中心算法的读数，判定默认椭圆中心是否可用。

```powershell
# 段 1（差分刀口）
.venv/Scripts/python.exe scripts/slm/issue235_beam_centre_differential_20261117.py `
  --out data/debug/issue235_center_20261117 --wait-until-idle 600 `
  --exposure-ms 0.05 --n-scan 41 --samples 5 --execute
```

- 差分刀口：清左半 − 清右半，checker 遮挡基座自动抵消。脚本本身已输出
  `zero_order_cy_cx`（argmax）与 `zero_order_box_centroid_cy_cx`（框内质心），
  并把 `flat.npy` 落盘。
- 离线补全四种口径（不占设备）：

```powershell
.venv/Scripts/python.exe -c "import numpy as np; from ao_shaping.utils.image.zero_order import zero_order_center_by_method as z; f=np.load('data/debug/issue235_center_20261117/flat.npy'); [print(m, z(f, m)) for m in ('argmax','centroid','box_centroid','ellipse')]"
```

- **判据（按坐标系分开，不可混减）**：`flat.npy` 的中心是 **CCD 坐标**，
  差分刀口中心是 **panel 坐标**，两者不可直接相减。因此：
  1. CCD 侧四口径两两之差 ≤ 3 px → 椭圆中心与既有口径等价，默认可用；
     差 > 10 px 说明半高连通域被杂散光/双瓣拉扯，报告里改述为"框内质心优先";
  2. panel 侧的刀口中心只与 `data/slm_geometry_calib.npz` 记录的历史值比
     （前值 x 889 / y 590），偏差 > 10 panel px 视为光路变动，需重写几何。
- 附带检查：`flat_peak_counts < 0.98 × flat_full_scale`；`background_median` 与
  历史同量级（~15 counts，8-bit / 差分路径）。

## 2. 阶段 P2 — 方向门槛（#235 / #234）

目标：在**放大奇响应**的条件下重测方向相关性——上一轮 0.03 rad 探针的奇/偶 rms 比
1.28–1.35 vs 模型 0.011–0.016，奇响应被约 40 倍偶分量淹没，这是门禁失去分辨力的直接原因。

- 探针：N4 / RX / RY 三对，`--probe-amplitude-rad` 依次跑 **0.03**（对照）与 **0.10**；
  0.10 rad 是 #234 校验档，也在本阶段顺带取证。
- 采集用 14 帧契约脚本（`A, A, N4±, A, RX±, A, RY±, A, A(0.5x), A(1.5x), A`），
  曝光 **0.1 ms**（10 ms 档会饱和，上一轮 20261117 会话即用 0.1 ms）：

```powershell
# 段 2（0.03 rad 对照档）
.venv/Scripts/python.exe scripts/slm/square_1000_model_diagnostic_20261030.py `
  --out data/debug/square1000_model_diag_20261117b_amp03 --wait-until-idle 600 `
  --base-exposure-ms 0.1 --probe-amplitude-rad 0.03 --execute

# 段 3（0.10 rad 放大档）
.venv/Scripts/python.exe scripts/slm/square_1000_model_diagnostic_20261030.py `
  --out data/debug/square1000_model_diag_20261117b_amp10 --wait-until-idle 600 `
  --base-exposure-ms 0.1 --probe-amplitude-rad 0.10 --execute
```

- 离线判读（两段都做，不占设备）：

```powershell
.venv/Scripts/python.exe scripts/slm/analyze_square1000_model_diagnostic_20261030.py --session data/debug/square1000_model_diag_20261117b_amp10
.venv/Scripts/python.exe scripts/slm/direction_peak_centred_20261117.py `
  --sessions data/debug/square1000_model_diag_20261117b_amp03 data/debug/square1000_model_diag_20261117b_amp10 `
  --out data/debug/direction_peak_centred_20261117b --center-method ellipse
```

- 判据（沿用 `jacobian_guard.direction_agreement`，阈值是先验、未标定）：
  - `correlation ≥ 0.50` 记 contradict 以上；
  - `≥ 0.85` 记 agree；
  - `rms < 2e-3 × mean|anchor|` 记 insufficient_signal，**不得**当失败计入分母。
- 只要 0.10 rad 档能把 cosine 抬到 ≥ 0.50，就证明"门禁失败 = 探针 SNR 不足"，
  而不是模型方向错；若仍 ≈ 0，则模型远场形态（PSF/defocus/照明包络）是主因。

## 3. 阶段 P3 — #234 结构化探针

- 幅度 {0.02, 0.05, 0.10} rad × ±号 × 重复，Noll 9 作 holdout。
- 帧预算：R=3 → 512 帧；R=1 → 176 帧（压缩档）。先跑 R=1，通过再加 R=3。
- 判据：per-mode `|corr| ≥ 0.5`、对角/非对角 ≥ 3、条件数 ≤ 50；
  `clamp_residual_block` 残差 L2 ≤ 0.5 rad。
- 模型切换闸：验证 NRMSE 退化 ≤ 10% 且方向守卫生效不下降；连续 2 次失败 → `rebuild_required`。

## 4. 阶段 P4 — #233 非线性取证

- 在 P3 的同一 session 里插一段**编码台阶扫描**：单一 Noll 模式，幅度按
  {0.005, 0.01, 0.02, 0.04, 0.08, 0.12} rad 递增，每档 5 帧，覆盖编码回读
  （`panel_gray` 反解）与实测强度的映射。
- 供电端到端非线性拟合使用，替代独立 gamma 标定；判据沿用 #233：
  freeform holdout correlation ≥ 0.8 且 NRMSE 低于 Zernike 族。
- 红线：均匀度结论只认 Mono12 且 plateau 落在量程中段；任何"预测变好"不算过。

## 5. 阶段 P5 — 关闭与记录

- 逐脚本核对 `released.json`、`device_log.json`；
- 回复中报告：占用设备、做了什么、何时释放、是否正常关闭；
- 数据落 `data/debug/`，分析结论落 `report/shape_trials/model_direction_hw_20261117/`；
- 每条结论同时回写 issue（#229 / #233 / #234 / #235）并附 commit id。

## 6. 失败处置

| 现象 | 处置 |
|---|---|
| 驱动板无应答 / `ReadSU=-1000` | 停手，给控制器断电重启；软件无法恢复，等冷却后重来 |
| `serial_source` 不是 driveboard | 视为探针无效，不写结论 |
| 峰值触顶 0.98 满量程 | 降曝光重拍该档，不裁数据 |
| 某阶段 `--out` 目录已存在 | 换新目录（transfer capture 目录不可复用） |
