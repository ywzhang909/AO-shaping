# 离线损失函数比较（20261107，[issue #236](https://github.com/ywzhang909/AO-shaping/issues/236)）：方向中位 + 核心61 Pearson

<!-- provenance:start -->
> **生成脚本**: [`scripts/slm/generate_loss_compare_report_20261107.py`](../../../scripts/slm/generate_loss_compare_report_20261107.py)
> **复现命令**: `python scripts/slm/generate_loss_compare_report_20261107.py`
> **运行环境**: 离线渲染（实验数据 2026-11-07：Santec #1 1064 nm + Daheng CCD 成对 U/C 存档）
> **说明**: #236 拟合损失比较（phase_sign=−1 主变体 + +1 历史对照；基线复现 FAILED，结论未定）：输入为 data/debug/slm_transfer/loss_compare_20261107_phase_sign_m1/results.json 存档产物，表格数字全部渲染时读取、零硬编码。报告本体不占用设备。
<!-- provenance:end -->

**日期** 2026-11-07 · **脚本** [`scripts/slm/generate_loss_compare_report_20261107.py`](../../../scripts/slm/generate_loss_compare_report_20261107.py)（离线渲染，未占用设备）
**数据** `data/debug/slm_transfer/loss_compare_20261107_phase_sign_m1/results.json`（2026-11-07T12:11:32+00:00 生成，`device_occupied=false`）· 数据集 `defocus_forward_20261107_focus_r2`：52 行清单 / 43 样本（25 训练 / 18 测试）/ 9 个 held-out U/C 对 / 121x121 裁剪 / 曝光归一

## 1. 背景与首轮作废

- [issue #236](https://github.com/ywzhang909/AO-shaping/issues/236) 为 `fit_loss` 重构做候选损失 对比择优：7 种损失（pearson / mse / log_mse / rel_l2 / masked_mse / ee_uniformity / composite）在冻结前向、只优化 (d, origin, log_gain) 三参数（Adam(d lr=0.01, origin lr=1.0, log_gain lr=0.05 when scale-sensitive)，300 步）下，于同一 U/C 回放数据上离线比较（[`loss_compare_20261107.py`](../../../scripts/slm/loss_compare_20261107.py)）。
- `composite` 为预登记权重组合（masked_mse 0.4 / pearson 0.4 / ee_uniformity 0.2）；初始 d=0.0802、origin=(1300.60, 943.22)、log_gain=0。
- **首轮作废**：首轮曾以 `+1` 极性为主变体、据下表历史对照推荐 `masked_mse`（方向中位 +0.483）。[#235](https://github.com/ywzhang909/AO-shaping/issues/235) 的单模屏配对证据（42/42 对在 `phase_sign=-1` 下方向相关全部翻正）推翻 `+1` 极性，首轮推荐随之作废；本报告为 `−1` 主变体重跑结果。

## 2. −1 主变体排序（sign_mode=negated）

口径（存档 JSON `ranking`，预登记排序：方向中位 ↓、核心61 ↑、FWHM 误差 ↑；仅 full_batch 变体，多 seed 平均）：

- **方向中位（符号 Pearson）**：每个 held-out U/C 对取 C−U 差分帧、去均值后全窗（121x121）Pearson，9 对取中位数，再对 seed 取均值；
- **核心61 Pearson**：测试帧核心 61×61 窗 Pearson 对测试帧取均值；
- **FWHM 误差（px）**：|预测面积 FWHM − 实测面积 FWHM| 对测试帧取均值。

| rank | loss | sign_mode | 方向中位（符号 Pearson） | 核心61 Pearson | FWHM 误差（px） |
|---|---|---|---|---|---|
| 1 | `mse` | negated | +0.070 | 0.589 | 1.42 |
| 2 | `ee_uniformity` | negated | +0.058 | 0.160 | 5.05 |
| 3 | `masked_mse` | negated | +0.037 | 0.584 | 1.65 |
| 4 | `rel_l2` | negated | +0.036 | 0.584 | 1.67 |
| 5 | `log_mse` | negated | +0.029 | 0.195 | 22.53 |
| 6 | `pearson` | negated | -0.018 | 0.566 | 2.21 |
| 7 | `composite` | negated | -0.043 | 0.486 | 3.98 |

- **无区分度**：7 种损失的方向中位全部落在 -0.043…+0.070，最优 `mse` 仅 +0.070（核心61 0.589）；held-out 仅 9 个 U/C 对，既有数据不足以在损失之间做选择。

## 3. +1 历史对照（sign_mode=current，legacy，仅对照口径）

该极性已被 #235 硬件证据推翻（见[forward_model_polarity_20261107.md](../forward_model_polarity/forward_model_polarity_20261107.md)），下表数字不构成推荐依据；其高方向中位正是在被推翻的极性下取得的。

| rank | loss | sign_mode | 方向中位（符号 Pearson） | 核心61 Pearson | FWHM 误差（px） |
|---|---|---|---|---|---|
| 1 | `masked_mse` | current | +0.483 | 0.729 | 2.12 |
| 2 | `rel_l2` | current | +0.469 | 0.725 | 2.07 |
| 3 | `pearson` | current | +0.390 | 0.716 | 2.23 |
| 4 | `mse` | current | +0.388 | 0.718 | 2.23 |
| 5 | `composite` | current | +0.329 | 0.678 | 2.64 |
| 6 | `ee_uniformity` | current | +0.069 | 0.041 | 8.44 |
| 7 | `log_mse` | current | +0.003 | 0.213 | 22.16 |

## 4. 基线复现 FAILED

判据：以存档参考拟合 `data/debug/slm_transfer/defocus_forward_20261107_focus_r2_fit/forward_fit.json` 为基线，同数据同口径重拟合（−1 变体），|Δd| ≤ 0.005 且 origin 偏移 ≤ 1 px。

| 量 | 存档基线 | −1 变体重拟合 |
|---|---|---|
| defocus d（D） | +0.1139 | -0.0021 |
| origin（px） | (1320.67, 945.61) | (1304.76, 944.45) |
| mean corr | 0.831（存档全行） | 0.661（测试全窗） |

- Δd = 0.116（容差 0.005）、origin 最大偏移 15.9 px（容差 1 px）→ `passed=false`。
- 旁证：当前 `phase_sign` 作用域为整条指令相位（含载波/倾斜项），整体取反会破坏这些非对称项——与 [#235](https://github.com/ywzhang909/AO-shaping/issues/235) 报告记录的自洽性矛盾一致（`−1` checkpoint 在同一 U/C 数据上全帧 Pearson 反而更低）。基线不可复现是损失比较结论不可靠的独立旁证。

## 5. 结论：未定

- 7 种损失无区分度（−1 主变体方向中位 -0.043…+0.070，held-out 仅 9 对）＋基线复现失败 → **无法据此选定损失函数**；
- [issue #236](https://github.com/ywzhang909/AO-shaping/issues/236) 的消融实现应等条件具备后重跑，重做条件：
  1. `phase_sign` 语义修复并通过单模屏方向门禁（单对 ≥ 0.50，[#235](https://github.com/ywzhang909/AO-shaping/issues/235) 门禁口径）；
  2. 在方向门禁通过后的数据上，以单模屏 / 闭环指标为主口径重做本比较（U/C 回放 9 个 held-out 对不足以支撑 7 路比较）。

## 6. 复现命令与产物

- 数据生成（已存档，勿重跑）：
  ```
  python scripts/slm/loss_compare_20261107.py \
      --output data/debug/slm_transfer/loss_compare_20261107_phase_sign_m1
  ```
- 报告渲染（离线，未占用设备）：
  ```
  python scripts/slm/generate_loss_compare_report_20261107.py
  ```
- 产物：`data/debug/slm_transfer/loss_compare_20261107_phase_sign_m1/results.json`（存档于 `data/debug/slm_transfer/loss_compare_20261107_phase_sign_m1/`）、本报告、数据集目录 `data/debug/slm_transfer/defocus_forward_20261107_focus_r2`、参考拟合 `data/debug/slm_transfer/defocus_forward_20261107_focus_r2_fit/forward_fit.json`。
- 全程离线（`device_occupied=false`），未占用 SLM / 相机 / 激光。
