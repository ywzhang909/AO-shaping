# 硬件窗口 2026-11-20b：#247 几何采集 + #238 周期扫描（真实上机）

用户授权窗口（2026-11-20 21:2x–）。本轮**真实开设备**，与 `issue238_issue247_capture_scripts_20261120.md`（纯离线车道）区分。目录 `data/debug/hw_window_20261120b/`（唯一新 `--out`，`mkdir(exist_ok=False)`）。

## 0. 设备与纪律

- 占用设备：Santec SLM-200 #1（1064 nm，`two_pi_grayscale=993`，memory 模式开）+ Daheng MER2-507-23GM（12 bit，全幅 2592×1944 不物理开窗）。未用 WFS / DM / 激光功率调节。
- 每段自取 `DeviceLock.default()`（OS 字节锁；`data/debug/.device.lock` 的 1 字节陈旧残留不阻塞，见 `device_mutex.py` 互斥区说明），带 `--wait-until-idle`。
- 开始前核对：系统内 python 进程均为 `mcp-server-time` / `mcp-server-fetch`，**无设备占用进程**。
- 每段 `finally`：`set_grayscale(0) → cam.close → slm.close → lock.release`；`released.json` 记 `closed_normally`。
- 段间留 settle / 冷却，避免背对背 open/close 触发驱动板假死。

## 1. 六步光路探测（`probe_optical_path_validity.py`，s0）

- 输出 `optical_path_probe/s0/20261120_132055Z/optical_path_record.json`，exit 0，`判定: ok`。
- 曝光阶梯（flat）峰值在 0.05 ms 达 1717 counts（≈42% 满量程），≤0.05 ms 全程未饱和；噪声底中位 15 + 4σ ≈ 20.9 counts。
- 离焦响应显著（宽度 +23.5%、峰值 +29.3%）⇒ 光路有效、SLM 确在调制。
- 采用单一固定曝光 **0.05 ms** 供后续两段。

## 2. #247 几何采集（`issue247_geometry_capture_20261120.py --execute`）

- 命令：`--execute --out data/debug/hw_window_20261120b/issue247_geometry --exposure-ms 0.05 --settle-s 0.8 --saturation-fraction 0.9`。
- 结果：`summary.status=ok`，**44 帧 / 11 图样**落盘（`.npy` 全幅 uint16 1944×2592），`saturation_count=0`，`released.json: closed_normally=true, lock_released=true`，耗时 38.8 s。
- 记录缺口（待脚本补）：`optical_path_record.json` 中 `drive_board_id`/`option_board_id`/`exposure_range_ms`/`gain`/`git_commit`/`occupancy.released_at` 均为 `null`（释放信息在 `released.json`）。**未调用 `require_drive_board()` 记录板 ID**；但图样清晰衍射，证明驱动板确在调制、非选板静默回落。

### 2.1 离线判读（读原始帧，非脚本自述）

四帧取均值、`gaussian_filter` 平滑后取 2D 局部极大 + 全帧行/列能量投影双验证。中心 0 级 `(x≈1228, y≈1104)`（与 `slm_geometry_calib.npz` 的 `center=(1104.8,1224.4)` 行列换序一致）。

| 图样（相位沿） | 1 级位移 | 3 级位移 | 1/周期验证 |
|---|---|---|---|
| grating_x p128（面板**列**） | ±95/96 px（CCD x） | ±284/285（3×95） | — |
| grating_x p256（面板列） | ±48 px（CCD x） | ±142/143（3×48） | 96/48 = 2.0 ✓ |
| grating_y p128（面板**行**） | ±60 px（CCD x） | ±177/178（3×59） | — |
| grating_y p256（面板行） | ±29/30 px（CCD x） | ±89/90（3×30） | 60/30 = 2.0 ✓ |

- **衍射级间距严格 ∝ 1/周期**（两轴各自 128→256 均精确 2.0，且 3 级 ≈ 3×1级）⇒ 光栅探针本身可靠，可直接反推 panel→CCD 比例（若能定轴）。
- **两面板轴比例各向异性**：同周期 128 下，面板列 96 px vs 面板行 60 px（比 1.6），与旧标定 `Kx≈Ky≈7555`（各向同性）不符。

| 图样 | 主能量方向 | 级位置（相对 0 级） |
|---|---|---|
| ladder_x（面板列） | CCD **y** | dy = −88, −232（单侧） |
| ladder_y（面板行） | CCD **x** | dx = +78, +235（单侧） |
| tilt_x(+) / (−) | CCD x | +95 / −96（反向 ✓） |
| tilt_y(+) / (−) | CCD x | +62 / −65（反向 ✓） |

- **单向 blaze 光斑梯单侧成立**：两轴均只见同侧阶（无镜像/反对称侧），`[0,+1,+3]` 结构可区分 180° 分支。
- **倾斜正负反向成立**：x、y 两轴 plus/minus 位移符号相反、幅度近等 ⇒ 手性/符号可判。

### 2.2 未解——面板↔CCD 轴映射不能冻结

6 条方向性探针中 4 条支持「面板行→CCD x、面板列→CCD y」的 **~90° 旋转**（grating_y 行→x、ladder_x 列→y、ladder_y 行→x、tilt_y 行→x）；但 **grating_x 与 tilt_x（面板列）却落在 CCD x**，与旋转模型矛盾。所有探针方向均严格轴对齐（无 45° 倾斜），故非简单刚体旋转，也非各向同性缩放。

- 结论：**当前数据不足以冻结物理映射**；与既有 #247 判定一致（缺 `slm_pixel_pitch_um` / `camera_pixel_size_um` / `focal_magnification`，plan `missing` 明列）。需补：面板 pitch 与 CCD 像元/焦距放大率的实测常量，或一张专用配准图样（如正交十字 + 已知偏移标记）以定轴与各向异性比。
- 不宣称新比例/旋转常数；不写 `slm_geometry_calib.npz`。

## 3. #238 周期扫描（`run_issue238_period_sweep_20261119.py --execute`）

- 命令：`--run-dir data/debug/hw_window_20261120b/issue238_sweep --periods-px 32 48 --period-order palindrome --frames-per-level 4 --power-reference abba --wait-until-idle 900 --execute`。
- 四段回文 `period32 → period48 → period48_r2 → period32_r2`，每段**全部** `status=ok` / `exit=0` / 段末锁 `free`，段间冷却 200 s。总耗时约 28.96 min（本地时间 `2026-11-20 21:31:02+08:00 → 21:59:59+08:00`）。
- 每段 205 帧（`4 bracket + 1 reference + 4 power_ref + 192 sweep + 4 drift`），55 图样 ×4 帧；固定曝光 **0.08 ms**（bracket 0.02/0.04/0.08 峰值 795/1539/2557 全可用，0.16 ms 饱和 62–83 px 被 `analysis_policy` 排除）。**192 个 sweep 数据帧零饱和**（唯一饱和帧是 bracket_03，已排除）。
- 2π 口径：`g_ref=1023`（编码标度）、`device_two_pi_gray_readback=993`（设备物理回读），`two_pi_gray_true="pending offline fit"`，符合 `two_pi_grayscale_source_decision_20261119.md`。
- 设备释放：各段 `release.json` 均 `slm_flat_gray0=ok / slm_closed=ok / cam_closed=ok / lock_released=ok / error=null`；窗口 `released.json` 记 `window_held_lock=false / final_state.status=free / holder=null`。

### 3.1 ABBA 功率漂移解耦（关键结果）

| 段 | 周期内 block | `drift_ratio_post_over_pre` | 段内帧 scale 范围 |
|---|---|---|---|
| period32 | block0_p32 | +1.000187 | [0.999871, 0.999897] |
| period48 | block0_p48 | −0.000696 | 近 1.0 |
| period48_r2 | block0_p48 | −0.000857 | 近 1.0 |
| period32_r2 | block0_p32 | +1.000399 | 近 1.0 |

- **run4 的 ±31% 跨扫漂移未复现**：本轮每段前后参考均值变化 <0.1%，帧 scale ≈1.0。故本轮数据与 run4 不同口径，**不可与 run4 的 g₂π 直接合并**，但可作为低漂移的独立机制复核。
- 每段仅 4 帧/灰度（run4 为 20 帧/灰度），统计精度低，定位为**机制检查**而非正式 g₂π 重测。

### 3.2 离线描述性判读（`data/debug/hw_window_20261120b/issue238_sweep/analysis/`）

纯离线读原始帧（不占用设备），用 ABBA 每段 `scale` 归一，0 级 `(x≈1227,y≈1105)`，±1 级位移 p32 ≈ ±236 px、p48 ≈ ±157 px（∝1/周期，与 #247 光栅测距一致）。图 `issue238_order_efficiency_20261120.png`，逐灰度数值 `issue238_descriptive.json`。

| 段 | ±1 峰值灰度 | +1 极小灰度 | −1 极小灰度 | 备注 |
|---|---|---|---|---|
| period32 | 448 | 952 | 944 | 4 帧/灰度 |
| period48 | 448 | 960 | 944 | 4 帧/灰度 |
| period48_r2 | 448 | 952 | 936 | — |
| period32_r2 | 448 | 960 | 944 | — |

- **地标峰 ≈448 灰度高度重现**（run1 地标峰也≈448），±1 极小落在 **936–960**，与 run1 系统包络 [952,981] 部分重叠、整体偏低约 1.5%，与 run4 折叠 946.9/952.6 方向一致；重复段同灰度 η₁ 比值 0.96–1.01（漂移归一后）。
- p48 的 η₁ 幅值约为 p32 的 1.35×（光栅占空/像素响应差异，属描述性观察，不进入 g₂π）。
- **边界**：本次**未采集暗场**（帧组只有 bracket/reference/power_ref/sweep/drift，无 dark），不满足计划 §6.6「永久报告门槛」，**不冻结新 g₂π**、不更新 `slm_lut.md`、不改 AGENTS 指针；4 帧/灰度的 CI 也不足以替代 run4 的 20 帧/灰度正式估计。

## 4. 判定

- **#247**：几何探针确认级间距 ∝1/周期、单侧 blaze 梯、tilt 符号反向；但面板↔CCD 轴映射仍存在 grating_x/tilt_x 与其余 4 条探针的矛盾，且缺 pitch/放大率常量，**维持 OPEN 的物理映射结论**，不冻结常数、不写 `slm_geometry_calib.npz`。与历史 `report/shape_trials/issue247_zero_order_20261117/issue247_zero_order_20261117.md` 的 +88.54°（180° 对称族）一致但本轮不独立确认。
- **#238**：ABBA 方案有效消除 run4 的漂移共线问题；低漂移复采的地标（448）、极小（936–960）与 run4/run1 粗略吻合，机制层面支持二值光栅破简并路线。因无暗场 + 仅 4 帧/灰度，**不满足永久报告门槛，不发布新 g₂π**，维持 OPEN 待正式 20 帧/灰度 + 暗场复核。
- 设备与锁全程正常释放；未 push、未改 GitHub issue 状态。
