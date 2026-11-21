# 湍流整形对比择优矩阵 (epochs=60, cam=256, n_max=4, side=120)

<!-- provenance:start -->
> **生成脚本**: [`scripts/slm/run_turb_shaping_matrix.py`](../../../scripts/slm/run_turb_shaping_matrix.py)
> **复现命令**: `python scripts/slm/run_turb_shaping_matrix.py --epochs 60 --out-root data/turb_matrix --report-root report/turbulence_shaping/matrix_20261121`
> **数据/关联脚本**: [`scripts/slm/generate_turbulence_shaping_report.py`](../../../scripts/slm/generate_turbulence_shaping_report.py)
> **运行环境**: 离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）
> **说明**: 3 变体 x 3 档湍流 (cn2=1e-14/2e-13/2e-12, distance=500 m) 对比择优矩阵汇总; 轴为 2026-11-21 实测 σ 标定后冻结 (~0.06/0.27-0.28/0.88-0.90 rad)
<!-- provenance:end -->

| 变体 | cn2 | 实测 σ_total [rad] | 光学求值 | 单元报告 |
|---|---|---|---|---|
| spgd | 1e-14 | 0.0639 | 124 | [report.md](spgd_cn21e-14/report.md) |
| square | 1e-14 | 0.0626 | 121 | [report.md](square_cn21e-14/report.md) |
| heuristic | 1e-14 | 0.0634 | 736 | [report.md](heuristic_cn21e-14/report.md) |
| spgd | 2e-13 | 0.2733 | 124 | [report.md](spgd_cn22e-13/report.md) |
| square | 2e-13 | 0.2779 | 121 | [report.md](square_cn22e-13/report.md) |
| heuristic | 2e-13 | 0.2848 | 736 | [report.md](heuristic_cn22e-13/report.md) |
| spgd | 2e-12 | 0.8836 | 124 | [report.md](spgd_cn22e-12/report.md) |
| square | 2e-12 | 0.8952 | 121 | [report.md](square_cn22e-12/report.md) |
| heuristic | 2e-12 | 0.8847 | 736 | [report.md](heuristic_cn22e-12/report.md) |

## 口径

- dynamic 时变屏 (逐光学求值独立), 幅度**实测** (stats), 见各单元报告;
- 收敛快慢/终值对比见各单元 GIF 与 metrics.png; 本矩阵为**对比择优**,
  MIL / GS-refine 因 sim 孪生无湍流通道暂不入列 (#253 注记);
- 种子: dist_seed = 20261121 + 单元序号 (左到右, 逐档递增)。
