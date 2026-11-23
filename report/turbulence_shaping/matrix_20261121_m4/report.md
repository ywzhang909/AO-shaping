# 湍流整形对比择优矩阵 (epochs=120, cam=256, n_max=11, side=120)

<!-- provenance:start -->
> **生成脚本**: [`scripts/slm/run_turb_shaping_matrix.py`](../../../scripts/slm/run_turb_shaping_matrix.py)
> **复现命令**: `python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --n-max 11 --variants spgd,heuristic --out-root data/turb_matrix_m4 --report-root report/turbulence_shaping/matrix_20261121_m4`
> **数据/关联脚本**: [`scripts/slm/generate_turbulence_shaping_report.py`](../../../scripts/slm/generate_turbulence_shaping_report.py)
> **运行环境**: 离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）
> **说明**: M4 拐点扫描汇总 (n_max=11, 120 epochs; spgd x heuristic 两变体 x 三档, 与 M3/matrix_20261121 同轴同种子): spgd n8→n11 无增益 (拐点=8); heuristic-PSO 提维单调劣化三档复现 (固定种群 12)
<!-- provenance:end -->

| 变体 | cn2 | 实测 σ_total [rad] | 光学求值 | 单元报告 |
|---|---|---|---|---|
| spgd | 1e-14 | 0.0635 | 244 | [report.md](spgd_cn21e-14/report.md) |
| heuristic | 1e-14 | 0.0622 | 1456 | [report.md](heuristic_cn21e-14/report.md) |
| spgd | 2e-13 | 0.2771 | 244 | [report.md](spgd_cn22e-13/report.md) |
| heuristic | 2e-13 | 0.2795 | 1456 | [report.md](heuristic_cn22e-13/report.md) |
| spgd | 2e-12 | 0.8963 | 244 | [report.md](spgd_cn22e-12/report.md) |
| heuristic | 2e-12 | 0.9058 | 1456 | [report.md](heuristic_cn22e-12/report.md) |

## 口径

- dynamic 时变屏 (逐光学求值独立), 幅度**实测** (stats), 见各单元报告;
- 收敛快慢/终值对比见各单元 GIF 与 metrics.png; 本矩阵为**对比择优**,
  MIL / GS-refine 因 sim 孪生无湍流通道暂不入列 (#253 注记);
- 种子: dist_seed = 20261121 + 单元序号 (左到右, 逐档递增)。
- best-of-N 选择偏置口径 (TODO 第 181 批注记): dynamic 时间白目标下 best 随求值次数单调向上偏置,
  heuristic 1456 求值 (≈6× spgd 244) 的 best 因**多求值变体**选择而高估, 与 spgd 的 best
  不宜按 6× 求值成本直接对比 (拐点/择优终判据此以 spgd 244 求值为基准)。
