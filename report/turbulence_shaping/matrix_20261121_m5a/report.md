# 湍流整形对比择优矩阵 (epochs=120, cam=256, n_max=4, side=120, pop=12)

<!-- provenance:start -->
> **生成脚本**: [`scripts/slm/run_turb_shaping_matrix.py`](../../../scripts/slm/run_turb_shaping_matrix.py)
> **复现命令**: `python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --variants gs_refine --out-root data/turb_matrix_m5a --report-root report/turbulence_shaping/matrix_20261121_m5a`
> **数据/关联脚本**: [`scripts/slm/generate_turbulence_shaping_report.py`](../../../scripts/slm/generate_turbulence_shaping_report.py)
> **运行环境**: 离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）
> **说明**: M5a/M5b 汇总: GS-refine freeform (12x12 全频网格) 湍流矩阵三档 (gs_refine 臂自带孪生湍流通道, 测量走 SimPibCCD 真光路); 高 σ 末值不降反为三档最高, Zernike square 的高 σ 退化在 freeform 下消失
<!-- provenance:end -->

| 变体 | cn2 | 实测 σ_total [rad] | 光学求值 | 单元报告 |
|---|---|---|---|---|
| gs_refine | 1e-14 | 0.0633 | 362 | [report.md](gs_refine_cn21e-14/report.md) |
| gs_refine | 2e-13 | 0.2783 | 362 | [report.md](gs_refine_cn22e-13/report.md) |
| gs_refine | 2e-12 | 0.9111 | 362 | [report.md](gs_refine_cn22e-12/report.md) |

## 口径

- dynamic 时变屏 (逐光学求值独立), 幅度**实测** (stats), 见各单元报告;
- 收敛快慢/终值对比见各单元 GIF 与 metrics.png; 本矩阵为**对比择优**,
  MIL / GS-refine 因 sim 孪生无湍流通道暂不入列 (#253 注记);
- 种子: dist_seed = 20261121 + 冻结网格序号 (``slot_of(variant, cn2)``; 子扫掠与基线逐单元配对)。
