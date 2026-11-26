# 湍流整形对比择优矩阵 (epochs=120, cam=256, n_max=11, side=120, pop=12)

<!-- provenance:start -->
> **生成脚本**: [`scripts/slm/run_turb_shaping_matrix.py`](../../../scripts/slm/run_turb_shaping_matrix.py)
> **复现命令**: `python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --n-max 11 --variants spgd,heuristic --out-root data/turb_matrix_m4b --report-root report/turbulence_shaping/matrix_20261121_m4`
> **数据/关联脚本**: [`scripts/slm/generate_turbulence_shaping_report.py`](../../../scripts/slm/generate_turbulence_shaping_report.py)
> **运行环境**: 离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）
> **说明**: M4 拐点扫描汇总 (n_max=11, 120 epochs; spgd x heuristic 两变体 x 三档, 与 M3/matrix_20261121 同轴): spgd n8→n11 增益不超出档间波动 (拐点=8); heuristic-PSO 提维劣化三档复现 (固定种群 12); 本目录为 f5e7550a 冻结槽位契约后的**配对重跑版** (data/turb_matrix_m4b, 6 单元与基线/M3 逐单元同 dist_seed, 首值逐位同可配对; 早期未配对版仅 spgd 1e-14 slot 0 恰好配对, 其结论方向不变、幅度以本版为准)
<!-- provenance:end -->

| 变体 | cn2 | 实测 σ_total [rad] | 光学求值 | 单元报告 |
|---|---|---|---|---|
| spgd | 1e-14 | 0.0635 | 244 | [report.md](spgd_cn21e-14/report.md) |
| spgd | 2e-13 | 0.2699 | 244 | [report.md](spgd_cn22e-13/report.md) |
| spgd | 2e-12 | 0.9028 | 244 | [report.md](spgd_cn22e-12/report.md) |
| heuristic | 1e-14 | 0.0632 | 1456 | [report.md](heuristic_cn21e-14/report.md) |
| heuristic | 2e-13 | 0.2864 | 1456 | [report.md](heuristic_cn22e-13/report.md) |
| heuristic | 2e-12 | 0.8939 | 1456 | [report.md](heuristic_cn22e-12/report.md) |

## 口径

- dynamic 时变屏 (逐光学求值独立), 幅度**实测** (stats), 见各单元报告;
- 收敛快慢/终值对比见各单元 GIF 与 metrics.png; 本矩阵为**对比择优**,
  MIL 因解析孪生 (`_SimBench` 钉死 cn2=0) 无湍流通道暂不入列 (GS-refine 已入列,
  见 `matrix_20261121_m5a`; #253 注记);
- 种子: dist_seed = 20261121 + 冻结网格序号 (``slot_of(variant, cn2)``; 子扫掠与基线逐单元配对)。
- **配对重跑版** (f5e7550a 槽位契约后, 6 单元与基线/M3 同 dist_seed, 首值逐位相同):
  spgd n_max 8→11 best 逐档 -0.793→-0.840 / -0.807→-0.792 / -0.796→-0.799,
  **无一致增益, 拐点=8 在配对口径下成立**; heuristic-PSO 劣化配对复现, 高 σ (2e-12)
  best 随维度 n4 -0.518 → n8 -0.621 → n11 **-0.840** 单调劣化 (固定种群 12 升维采样不足)。
- best-of-N 选择偏置口径 (第 181 批注记): dynamic 时间白目标下 best 随求值次数
  单调向上偏置, heuristic 1456 求值 (≈6× spgd 244) 的 best 因**多求值变体**选择而高估,
  不宜按 6× 求值成本与 spgd 直接对比 (拐点/择优终判以 spgd 244 求值为基准)。
