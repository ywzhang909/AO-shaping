# phase_sign 作用域实验（issue #235）— 离线结果

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **运行环境**: 离线回放存档数据（2026-11-07；未占用 SLM/相机/激光，未 import ao_shaping.drivers）
> **说明**: #235 phase_sign 作用域实验：四变体（command/aberration/aberration_focus/aberration_carrier）离线拟合与单模屏方向对照；command 精确复现存档 −1（Pearson 路径 Δd 4.7e-10）；屏方向四变体等价、无法区分作用域；aberration_focus 不稳定需复现；carrier 不可分离。输入为 data/debug/slm_transfer/phase_scope_20261107/ 存档产物。人工撰写。
<!-- provenance:end -->

> **数据** [`results.json`](../../../data/debug/slm_transfer/phase_scope_20261107/results.json) · 详细过程 [`REPORT.md`](../../../data/debug/slm_transfer/phase_scope_20261107/REPORT.md) · 运行日志 [`run.log`](../../../data/debug/slm_transfer/phase_scope_20261107/run.log)

## 结论摘要

- 作用域机制可用：`command` 变体精确复现存档 −1 拟合（Pearson 路径，Δd 4.7e-10，origin 偏差 0 px）。
- 单模屏方向评估：四个作用域数值等价（中位数相同，均值差约 1e-4），**无法据此区分作用域**。−1 符号翻转幅度与存档 +1 屏互为镜像（amp 0.05：−0.496 ↔ +0.496）。
- `aberration_focus` 变体使 d 翻号（+0.024914 → −0.115916）、origin 偏移 20.6 px，**结果不稳定，需复现**。
- `aberration_carrier` 不可分离，等同 `aberration`（未伪造差异）。
- 非零作用域 checkpoint 导出往返通过（pearson ≥ 0.999998 对拟合预测）。

## 四变体对照（M1_defocus）

| 作用域 | d | origin (x, y) | Δd vs command | origin 偏差 |
|---|---|---|---|---|
| command | +0.002432 | (1305.56, 944.53) | — | — |
| aberration | +0.024914 | (1308.68, 944.86) | +0.022482 | 3.12 px |
| aberration_focus | −0.115916 | (1284.96, 943.56) | −0.118348 | 20.60 px |
| aberration_carrier | +0.024914 | (1308.68, 944.86) | 同 aberration | 同 aberration |

基线口径：回归护栏采用 Pearson 路径（`loss_compare_20261107`），`fit_defocus_forward_dataset.py` 给出的 d=0.002432 作为独立字段记录，HEAD 上即存在该差异，非本次重构引入。

focus 仅在 C 行包含（`focus_overlay_max_abs_diff` ≈ 2.4e-7），仅 `aberration_focus` 对 C 行施加 −2F，**不计入总相位**；本分解只影响 C 行。

## 单模屏方向（`--origin model`，无配准，n=14/组）

| 组 | 作用域 | 中位数 | 均值 | std | ≥0.5 |
|---|---|---|---|---|---|
| amp 0.10 | 四变体相同 | +0.441 | +0.399 | 0.191 | 4/14 |
| amp 0.05 | 四变体相同 | +0.496 | +0.484 | 0.169 | 7/14 |
| amp 0.02 | 四变体相同 | +0.459 | +0.469 | 0.159 | 6/14 |

存档 +1 对照：amp 0.05 中位 −0.496，amp 0.02 中位 −0.459（0/14 ≥0.5）；amp 0.10 的 +1 文件未找到同名存档，该格留空。

## 导出往返

四变体 checkpoint 经生产 loader 回读，与拟合预测比对：pearson 0.999998–0.999999，rel L2 1.3e-3–2.5e-3。`aberration_focus` 以 `(phase − F) + F` 重放（F 为包含量）。

## 待办 / 未决

- 基线选择已由用户确认为 Pearson 路径；`fit_defocus_forward_dataset.py` 与其差异仍待单独溯源。
- U/C 分支级 Δd 未计算。
- `aberration_focus` 需复现后再采信。
- `multishape_objective.py` 由其他通道修改，未纳入本次改动。
