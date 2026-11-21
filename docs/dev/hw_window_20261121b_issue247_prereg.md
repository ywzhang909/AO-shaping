# 设备窗口 2026-11-21b · #247 A1 复采 + 2-D 对角 blaze 预注册

> 状态：**采集前冻结稿**。本文写于开窗前，图样清单、参数、目录与判据一律不得事后改动。
> 上游：`next_experiment_plan_and_capture_20261120.md` §3.3/§3.4/§8/§9、
> `agent_notes_20261121.md` §2「下一步」。
> 本文不自行授权设备占用；执行必须以用户当次明确的上机批准为准。

## 1. 为什么这一次要复采

2026-11-21 上午的 A1 采集（`data/debug/hw_window_20261121/issue247_axis/`，11 图样 × 4 帧）
得到 `frozen=false, verdict=mapping_still_ambiguous`：阶梯 family 与历史 88.544° 旋转自洽，
而 grating/tilt 两族把**两个正交 panel 轴都映到 CCD 列方向**（`tilt_x` vs `tilt_y` 仅差 0.63°）。
任何可逆 2×2 映射都做不到这一点，因此现有图样无法判定是"探针口径问题"还是"映射本身"。

2-D 对角 blaze 用一次曝光把一个**非对称**对角点成像到远场：该点的位置同时编码两轴符号与标度，
且对 axis swap、单轴符号翻转、rot180 三者都不等价。它是这批图样中唯一能把矛盾收敛到二维的探针。

同时复采一遍既有 11 图样，用于 `grating_x_p128` 等 ±列结果的稳定性（同名同参第二次独立采集）。

## 2. 冻结图样清单（15 图样 × 4 帧 = 60 帧）

| # | id | kind | 关键参数 |
|---|---|---|---|
| 1 | `flat` | flat | 0 级参考 |
| 2 | `grating_x_p128` | grating | period 128, axis=panel-x(cols) |
| 3 | `grating_y_p128` | grating | period 128, axis=panel-y(rows) |
| 4 | `grating_x_p256` | grating | period 256, axis=panel-x |
| 5 | `grating_y_p256` | grating | period 256, axis=panel-y |
| 6 | `ladder_x` | ladder | patch 600 px, base period 96, orders {1,3} |
| 7 | `ladder_y` | ladder | 同上 |
| 8 | `tilt_x_plus` | tilt | period 128, sign + |
| 9 | `tilt_x_minus` | tilt | period 128, sign − |
| 10 | `tilt_y_plus` | tilt | effective period 120, sign + |
| 11 | `tilt_y_minus` | tilt | effective period 120, sign − |
| 12 | `blaze2d_px128_py192_sx+1_sy+1` | blaze2d | fx=+1/128, fy=+1/192 |
| 13 | `blaze2d_px128_py192_sx+1_sy-1` | blaze2d | fx=+1/128, fy=−1/192 |
| 14 | `blaze2d_px192_py128_sx+1_sy+1` | blaze2d | fx=+1/192, fy=+1/128 |
| 15 | `blaze2d_px192_py128_sx+1_sy-1` | blaze2d | fx=+1/192, fy=−1/128 |

第 12–15 项是本窗新增。**两个周期对（128/192 与 192/128）都是必需的**：单靠 `sy` 符号翻转
只给出 −M·v，四次测量在拟合里线性相关，残差恒为 0；只有第二个独立方向才能让
"残差 RMS ≤1.0 px" 这条判据真正可检验。

全幅 2-D ramp 的能量全部进单一对角 1 级，DC 为数学零点；沿用既有 2.1 s/帧量级，60 帧约 2–3 min。

## 3. 冻结参数

| 项 | 值 |
|---|---|
| 脚本 | `scripts/slm/issue247_geometry_capture_20261120.py`（本窗扩展 blaze2d） |
| `--exposure-ms` | `0.05` |
| `--settle-s` | `0.8` |
| `--saturation-fraction` | `0.9` |
| 帧/图样 | 4（均值降噪，沿用 A1 口径） |
| 设备 | Santec SLM-200 #1（1064 nm，2π 物理灰度 993，编码标度 1023）+ Daheng MER2-507-23GM（12 bit 全幅 2592×1944，不物理开窗） |
| SLM 编号 | `--slm-number 1`，`--wavelength-nm 1064` |
| 在飞保护 | `--wait-until-idle 900` |

曝光 0.05 ms 的依据：同日光路探测可用曝光 0.05 ms（峰值 1834 counts、噪底 20.9），
本日上午 A1 平地峰值 1717–1923 counts（≈42–47%，未饱和）。blaze2d 与 `flat` 同为全孔径
单点，峰值量级应同阶；`--saturation-fraction 0.9` 的逐帧保护保留，一旦饱和即中止，不降低门限。

## 4. 目录（唯一新目录，`mkdir(exist_ok=False)`）

```
data/debug/hw_window_20261121b/
├── optical_path_probe/s0/<UTC>/optical_path_record.json   # 六步探测，段 0
└── issue247_axis/
    ├── frames/*.npy              # uint16, 1944×2592 全幅
    ├── summary.json              # 图样清单、phase sha256、饱和计数、blaze2d_designs、status
    ├── optical_path_record.json  # ao-optical-path-record/v1
    └── released.json             # closed_normally / lock_released
```

不得复用、不得覆盖 `data/debug/hw_window_20261121/`（上午那一轮是映射歧义结论的证据）。

## 5. 窗口顺序与释放纪律

| 顺序 | 段 | 预计 | 说明 |
|---|---|---|---|
| 0 | 六步光路探测 `probe_optical_path_validity.py --session issue247_20261121b --wait-until-idle 900` | ~2 min | verdict 必须为 `ok`；否则**中止本窗**，不采 A1 |
| 1 | 冷却 ≥20 s | — | 防背对背 open/close 触发驱动板假死（执行前按用户要求由 200 s 改为 20 s，见文末变更记录） |
| 2 | A1 采集（单次 `--execute`，单次 open/close） | ~3 min | 60 帧 |
| 3 | 解算 `issue247_axis_map_solve_20261121.py` | 离线 | 采集进程已释放设备后执行 |
| 4 | 释放核查 | — | 锁 mtime 刷新、无残留 python bench 进程 |

每段 `finally`：`set_grayscale(0) → cam.close → slm.close → lock.release`，并写 `released.json`。

> 判读纪律：`data/debug/.device.lock` 的 1 字节陈旧残留不阻塞（OS 字节锁无人持有），
> 但仍须先确认无 bench 进程占用。

## 6. 冻结判据（采集前写死，不得事后放宽）

解算模型：`(dr, dc) = A · (fy, fx)`，A 为 2×2（CCD px / panel 频率），只用第 12–15 项四点
做最小二乘（8 方程 / 4 未知量）。

`frozen = true` 当且仅当下列**全部**成立：

1. blaze2d 残差 RMS ≤ **1.0 px**；
2. 最大逐点相对误差 ≤ **2 %**（|residual| / |measured displacement|）；
3. 拟合 A 的两列方向夹角 = **88.544° ± 2°**；
4. 秩守卫 `|det A| / σ_max² ≥ 0.10`（排除"两轴塌到同一条 CCD 线"的退化解）。

任一条不满足 → `frozen=false`，`verdict` 指出失败条件，**不写** `issue247_axis_map.json`。

辅助证据（不参与冻结判定，仅用于归因）：grating-only / tilt-only / ladder-only 三族的
各自 2×2 残差，用于说明矛盾落在哪一族；既有方向角对照全部保留。

## 7. 不做什么

- 不写 `slm_geometry_calib.npz`；不把单次映射称作"物理常数已定"。
- 不在映射冻结前重判方向门（#234/#236 的双门 fail 不受本窗影响）。
- 不用本窗数据为 transfer `light/none` 解禁，也不作为 P4 翻案证据。
- 若探测 verdict ≠ `ok`，或任一段饱和中止：本窗作废，记录失败原因，不改判据重采。
- 不改 `closed_loop_shaping_page.py` 的串行语义；本窗不涉及 GUI。
- 不 push、不改 GitHub issue 状态（除非用户另行授权）。

## 8. 执行前变更记录

- 2026-11-21 17:38：段间冷却由 ≥200 s 改为 **≥20 s**，按用户明确要求。其余冻结参数、图样清单、
  目录与 §6 判据均不变。该变更仅影响段间等待，不影响任何采集口径或接受条件。
