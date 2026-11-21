# Issue #235 P1：零级中心四口径离线对账（2026-11-20）

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **运行环境**: 离线（仅读 data/debug/issue235_center_20261117/ 归档 flat.npy 与 issue235_knife_baseline_fit_20261117.json，未占用 SLM/CCD/激光）
> **说明**: Issue #235 P1 零级中心四口径离线对账：对归档 flat.npy 用生产函数 ao_shaping.utils.image.zero_order.zero_order_center_by_method 跑 argmax/centroid/box_centroid/ellipse 四法，六组两两欧氏距离最大 3.0 px（恰在 ≤3 px 判据上，边界通过），默认椭圆中心 (x=1227, y=1106) 可用（verdict=ellipse_centre_usable）；CCD 帧中心与 SLM panel 刀口帧中心（find_beam_on_slm_centre_yx / fit_centre_yx，1920×1200）分开报告、不可混减。数字来自同目录 reconciliation.json。无独立生成脚本，四法直接调用生产函数完成。人工撰写。
<!-- provenance:end -->

## 摘要

对归档的 `flat.npy`（`data/debug/issue235_center_20261117/`，2026-11-17 采集，本次不占设备）用生产函数
`ao_shaping.utils.image.zero_order.zero_order_center_by_method` 跑 `argmax` / `centroid` / `box_centroid` /
`ellipse` 四法求零级光斑中心。四法两两欧氏距离最大 **3.0 px**（`argmax` 与 `centroid`/`box_centroid`），
其余对不超过 2.0 px，全部满足 `hw_plan_consolidated_20261118.md` §3 P1 的「两两差 ≤3 px」判据。
默认椭圆中心 `(x=1227, y=1106)` 与其余三法差 1–2 px，**结论：椭圆中心可用**（`verdict=ellipse_centre_usable`）。
CCD 帧（2592×1944）与 SLM panel 刀口帧（1920×1200）坐标分开报告，**不可混减**。

## 口径与输入

- 判据来源：`docs/dev/hw_plan_consolidated_20261118.md` §3 P1 第 2 项——对 `flat.npy` 跑四法，两两差 ≤3 px
  则默认椭圆中心可用；CCD 侧与 panel 侧刀口中心分开比，不可混减。
- 四法实现：统一复用生产函数 `zero_order_center_by_method`（与 `tests/ao_shaping/utils/image/test_zero_order.py`
  测的是同一实现），避免四套临时口径。`argmax`=全局峰值；`centroid`=扣角点背景后全幅一阶矩；
  `box_centroid`=半高最大连通域外接框内加权质心；`ellipse`=半高最大连通域二阶矩中心（默认）。
- 输入数据（全部已在盘，离线）：
  - `data/debug/issue235_center_20261117/flat.npy`（1944×2592，float64，max=163.4 counts，未饱和）
  - `data/debug/issue235_center_20261117/summary.json`（归档的 argmax / box_centroid 值，用于交叉核对）
  - `report/shape_trials/model_direction_hw_20261117/issue235_knife_baseline_fit_20261117.json`（SLM panel 侧刀口中心）
- 运行环境：**离线**（仅读已保存 artefact，未占用 SLM / CCD / 激光）。

## 四法中心（CCD 帧，记法 cy,cx）

| 方法 | 中心 (cy, cx) | 与默认椭圆中心差 (px) |
|---|---|---:|
| `argmax`（最大值） | (1106, 1229) | 2.0 |
| `centroid`（质心） | (1106, 1226) | 1.0 |
| `box_centroid`（框内质心） | (1106, 1226) | 1.0 |
| `ellipse`（椭圆中心，默认） | (1106, 1227) | 0.0 |

## 两两差（欧氏距离，px，判据 ≤3.0）

| 对 | Δy | Δx | 距离 | ≤3 px |
|---|---:|---:|---:|:---:|
| argmax ↔ centroid | 0 | 3 | 3.0 | ✅ |
| argmax ↔ box_centroid | 0 | 3 | 3.0 | ✅ |
| argmax ↔ ellipse | 0 | 2 | 2.0 | ✅ |
| centroid ↔ box_centroid | 0 | 0 | 0.0 | ✅ |
| centroid ↔ ellipse | 0 | −1 | 1.0 | ✅ |
| box_centroid ↔ ellipse | 0 | −1 | 1.0 | ✅ |

- 最大两两差 **3.0 px**，恰在阈值上（边界通过）；椭圆中心与其余三法 1–2 px，一致性强。
- **边界说明**：归档 `summary.json` 的 `box_centroid` 用采集脚本内的本地 0.15 阈值固定窗口helper（浮点）
  得 `(1106.23, 1225.99)`，与生产函数（半高连通域后取整）的 `(1106, 1226)` 舍入后一致；
  若用归档浮点值直接与 argmax `(1106, 1229)` 相减，Δx≈3.008 px 会微超 3 px。本对账统一采用生产函数口径
  （四法同一实现、同取整）判据，故为 3.0 px 通过。此差异源于两处 helper 口径不同，已在 `reconciliation.json`
  的 `summary_crosscheck` 中如实登记。

## CCD 侧 / panel 侧刀口中心（分开报告，不可混减）

| 来源 | 中心 (y, x) | 坐标系 |
|---|---|---|
| `find_beam_on_slm_centre_yx` | (617.80, 882.00) | SLM panel，1920×1200 |
| `fit_centre_yx`（自由基线 erf 拟合） | (615.97, 887.14) | SLM panel，1920×1200 |

- 这两个是 panel 侧差分刀口的中心的两种估法（原始寻找 vs 拟合），彼此差 (1.82, 5.14) px。
- 它们与上文零级光斑中心 `(cy=1106, cx=1227)` **不在同一坐标系**（panel 1920×1200 vs CCD 2592×1944），
  按 §3 P1 要求不与其相减、不参与四法两两对账。

## 限制与结论

- 本对账仅覆盖 P1「四口径一致性」子项；`hw_plan_consolidated` §3 P1 的差分刀口采集本身此前已完成。
- 判据的「≤3 px 则默认椭圆中心可用」为归档计划既定口径，本次只执行判定，不新设或放宽阈值。
- 结论：四法两两差均 ≤3 px（最大 3.0 px，边界通过），默认椭圆中心可用；P1 四口径离线对账**已完成并落盘**。

## 复现与产物

本对账无独立生成脚本（溯源头已标注「人工撰写，无生成脚本」）；四法统一直接调用生产函数
`ao_shaping.utils.image.zero_order.zero_order_center_by_method`（与 `tests/ao_shaping/utils/image/test_zero_order.py`
同一实现）。最小复现：

```python
import numpy as np
from ao_shaping.utils.image.zero_order import zero_order_center_by_method

frame = np.load("data/debug/issue235_center_20261117/flat.npy")
for m in ("argmax", "centroid", "box_centroid", "ellipse"):
    x, y = zero_order_center_by_method(frame, m)   # 返回 (x, y)，即 (cx, cy)
```

刀口中心两值（`find_beam_on_slm_centre_yx` / `fit_centre_yx`）直接读取既有归档
`report/shape_trials/model_direction_hw_20261117/issue235_knife_baseline_fit_20261117.json`，
不做重算。产物：`report/shape_trials/issue235_zero_order_four_method_20261120/reconciliation.json`
（含四中心、六组两两差、判据、交叉核对与刀口中心）。
