# 湍流整形对比择优矩阵 (epochs=120, cam=256, n_max=8, side=120, pop=12)

<!-- provenance:start -->
> **生成脚本**: [`scripts/slm/run_turb_shaping_matrix.py`](../../../scripts/slm/run_turb_shaping_matrix.py)
> **复现命令**: `python scripts/slm/run_turb_shaping_matrix.py --variants model_in_loop --epochs 120 --n-max 8 --cam-size 256 --out-root data/turb_matrix_m5b --report-root report/turbulence_shaping/matrix_20261126_m5b`
> **数据/关联脚本**: [`scripts/slm/generate_turbulence_shaping_report.py`](../../../scripts/slm/generate_turbulence_shaping_report.py)
> **运行环境**: 离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）
> **说明**: M5b 汇总 (model_in_loop 三档, static 屏): MIL 孪生湍流通道首次入列 (`_SimBench` disturbance 注入, 孪生 pkl 行补 _c/_img); quality 首→末 0.665→0.756 (σ≈0.071) / 0.666→0.712 最优 0.744 (σ≈0.240) / 0.667→0.721 最优 0.753 (σ≈1.227), 三档末值全部优于平场; static 屏协议与 dynamic 各变体不可直接同表排序 (屏协议见口径); #253 择优终判不变 (dynamic 下 n_max=8 spgd-adamod)
<!-- provenance:end -->

| 变体 | cn2 | 实测 σ_total [rad] | 光学求值 | 单元报告 |
|---|---|---|---|---|
| model_in_loop | 1e-14 | 0.0709 | 2013 | [report.md](model_in_loop_cn21e-14/report.md) |
| model_in_loop | 2e-13 | 0.2398 | 2005 | [report.md](model_in_loop_cn22e-13/report.md) |
| model_in_loop | 2e-12 | 1.2269 | 2007 | [report.md](model_in_loop_cn22e-12/report.md) |

## 口径

- 屏协议: spgd/square/heuristic/gs_refine = dynamic (逐光学求值独立屏); model_in_loop = static (Step A 拟合与逐轮验收假设准静态波前, 见 cell_plan 注);
- 幅度**实测** (stats), 收敛快慢/终值对比见各单元 GIF 与 metrics.png; 本矩阵为**对比择优**; MIL 的孪生湍流通道自 M5b 入列 (`_SimBench` disturbance 注入; GS-refine 通道见 matrix_20261121_m5a; #253 注记);
- 种子: dist_seed = 20261121 + 冻结网格序号 (``slot_of(variant, cn2)``; 子扫掠与基线逐单元配对)。
