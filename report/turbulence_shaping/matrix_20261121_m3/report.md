# 湍流整形对比择优矩阵 (epochs=120, cam=256, n_max=8, side=120)

<!-- provenance:start -->
> **生成脚本**: [`scripts/slm/run_turb_shaping_matrix.py`](../../../scripts/slm/run_turb_shaping_matrix.py)
> **复现命令**: `python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --n-max 8 --out-root data/turb_matrix_m3 --report-root report/turbulence_shaping/matrix_20261121_m3`
> **数据/关联脚本**: [`scripts/slm/generate_turbulence_shaping_report.py`](../../../scripts/slm/generate_turbulence_shaping_report.py)
> **运行环境**: 离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）
> **说明**: M3 调参轮矩阵汇总 (n_max=8, 120 epochs, 与 matrix_20261121 基线 n_max=4/60ep 同轴同种子); 实测结论: spgd 三档全面大幅改善且对 σ 平坦, square 高 σ 退化修复, heuristic-PSO 提维反而变差
<!-- provenance:end -->

| 变体 | cn2 | 实测 σ_total [rad] | 光学求值 | 单元报告 |
|---|---|---|---|---|
| spgd | 1e-14 | 0.0635 | 244 | [report.md](spgd_cn21e-14/report.md) |
| square | 1e-14 | 0.0623 | 241 | [report.md](square_cn21e-14/report.md) |
| heuristic | 1e-14 | 0.0632 | 1456 | [report.md](heuristic_cn21e-14/report.md) |
| spgd | 2e-13 | 0.2699 | 244 | [report.md](spgd_cn22e-13/report.md) |
| square | 2e-13 | 0.2835 | 241 | [report.md](square_cn22e-13/report.md) |
| heuristic | 2e-13 | 0.2864 | 1456 | [report.md](heuristic_cn22e-13/report.md) |
| spgd | 2e-12 | 0.9028 | 244 | [report.md](spgd_cn22e-12/report.md) |
| square | 2e-12 | 0.8933 | 241 | [report.md](square_cn22e-12/report.md) |
| heuristic | 2e-12 | 0.8939 | 1456 | [report.md](heuristic_cn22e-12/report.md) |

## 口径

- dynamic 时变屏 (逐光学求值独立), 幅度**实测** (stats), 见各单元报告;
- 收敛快慢/终值对比见各单元 GIF 与 metrics.png; 本矩阵为**对比择优**,
  MIL / GS-refine 因 sim 孪生无湍流通道暂不入列 (#253 注记);
- 种子: dist_seed = 20261121 + 单元序号 (左到右, 逐档递增)。
