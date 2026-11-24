# 湍流整形对比择优矩阵 (epochs=120, cam=256, n_max=11, side=120, pop=36)

<!-- provenance:start -->
> **生成脚本**: [`scripts/slm/run_turb_shaping_matrix.py`](../../../scripts/slm/run_turb_shaping_matrix.py)
> **复现命令**: `python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --n-max 11 --pop-size 36 --variants heuristic --cn2-grid 2e-12 --out-root data/turb_matrix_m5c_n11p36 --report-root report/turbulence_shaping/matrix_20261121_m5c_n11p36`
> **数据/关联脚本**: [`scripts/slm/generate_turbulence_shaping_report.py`](../../../scripts/slm/generate_turbulence_shaping_report.py)
> **运行环境**: 离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）
> **说明**: M5c 汇总 (heuristic n11 pop36 @2e-12): 同 slot 配对; best -0.702 vs 11×12 的 -0.840 (3× 求值收复 0.138, 仍劣于 8×12 且远差于 4×12) → 种群重配不能翻案维度排序, 升维劣化主因是 PSO 搜索结构而非种群不足; 六行对照表见正文 M5c 节; #253 择优终判不变 (n_max=8 spgd-adamod)
<!-- provenance:end -->

| 变体 | cn2 | 实测 σ_total [rad] | 光学求值 | 单元报告 |
|---|---|---|---|---|
| heuristic | 2e-12 | 0.8972 | 4360 | [report.md](heuristic_cn22e-12/report.md) |

## 口径

- dynamic 时变屏 (逐光学求值独立), 幅度**实测** (stats), 见各单元报告;
- 收敛快慢/终值对比见各单元 GIF 与 metrics.png; 本矩阵为**对比择优**,
  MIL 因解析孪生 (`_SimBench` 钉死 cn2=0) 无湍流通道暂不入列 (GS-refine
  自带孪生湍流通道已入列, 见 matrix_20261121_m5a; #253 注记);
- 种子: dist_seed = 20261121 + 冻结网格序号 (``slot_of(variant, cn2)``; 子扫掠与基线逐单元配对)。

## M5c 种群重配结论 (配对口径, heuristic @2e-12 slot 8)

| n_max × pop | best shape | 光学求值 | vs 同维 pop12 |
|---|---|---|---|
| 4 × 12 (基线 matrix_20261121) | -0.518 | 1456 | — |
| 8 × 12 (M3) | -0.621 | 1456 | — |
| 8 × 24 (**本矩阵 n8p24 单元**) | **-0.632** | 2908 | +0.011 量级 ≈ 无增益 |
| 11 × 12 (M4/m4b 配对重跑) | -0.840 | 1456 | — |
| 11 × 36 (**n11p36 矩阵单元**) | **-0.702** | 4360 | 改善 0.138, 仍劣于 8×12 |

- 六行同 slot (dist_seed=20261129, 首值 -1.3574656828072158 逐位同) 可配对。
- **判定: 种群重配不能翻案维度排序** —— n8 种群 12→24 (2× 求值) 增益在噪声量级;
  n11 种群 12→36 (3× 求值) 只收复 0.138, 终值仍差于 8×12 且远差于 4×12。
  结合 best-of-N 多求值偏置口径 (第 181 批注记), 「升维单调劣化」主因是 PSO
  搜索空间结构而非种群不足。
- **#253 择优终判不变**: 湍流 dynamic 下「整形质量 × 求值成本」最优 = n_max=8
  spgd-adamod; M5c 候选就此收口。
